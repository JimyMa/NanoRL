"""DLSlime weight transport for M3.

Train side registers gathered tensors as RDMA memory regions, returns a
manifest. Rollout side allocates receive buffers, registers them under
the same names, pre-resolves local + remote handles, and issues a single
``endpoint.read(...)`` call against the underlying ``RDMAEndpoint``.

The lower-level direct path (``conn.endpoint.read(numeric_assign)``) is
used instead of ``peer_agent.read(named_assign)`` so we resolve handles
once at register time rather than on every call. For Qwen3-4B that's
~290 tensors per sync — the named-resolve overhead would otherwise
double per-call latency.

NIC selection follows ``available_nic()[local_rank % len(nics)]``: each
peer pins a single device, so multi-rank deployments spread across NICs
naturally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import torch

from dlslime import available_nic, PeerAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NIC selection
# ---------------------------------------------------------------------------


def select_nic(local_rank: int = 0, *, override: str | None = None) -> str:
    """Pick an RDMA NIC by ``local_rank % len(available_nic())``.

    Set ``override`` to bypass — useful when the caller already knows which
    NIC to use (e.g. matching the GPU's NUMA-local NIC by hand).
    """
    if override:
        return override
    nics = available_nic()
    if not nics:
        raise RuntimeError("no RDMA NICs available; cannot run weight transport")
    return nics[local_rank % len(nics)]


# ---------------------------------------------------------------------------
# Manifest types
# ---------------------------------------------------------------------------

_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


@dataclass
class TensorMRInfo:
    name: str  # logical (e.g. HF param name)
    mr_name: str  # RDMA region name (versioned, e.g. "weights:N:foo")
    size: int  # bytes
    shape: tuple[int, ...]
    dtype: str  # str(torch.dtype) value from _DTYPE_TO_STR


@dataclass
class WeightManifest:
    """Wire-form description of a versioned weight set.

    Sent train→rollout via SlimeRPC (or any out-of-band channel); contains
    everything the rollout side needs to allocate matching receive buffers
    and issue the RDMA read.
    """

    version: int
    train_alias: str
    entries: list[TensorMRInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Train side
# ---------------------------------------------------------------------------


class WeightTransportTrain:
    """Owns the lifecycle of versioned weight MRs on the train side.

    For each ``register(version, named_tensors)`` call we:
        * stage every tensor as a contiguous CPU buffer (CUDA tensors have
          to land on the host first; registering CUDA pointers is possible
          but introduces driver-version dependencies we'd rather not gate
          M3 on)
        * register one MR per tensor under a versioned name
        * keep the staged buffers alive on this object until ``unregister``
          to prevent the rollout from reading freed memory

    The returned ``WeightManifest`` is plain data; serialize it however the
    caller prefers (we ship via SlimeRPC in the actor wiring).
    """

    def __init__(self, peer_agent: PeerAgent, *, prefix: str = "weights"):
        self._agent = peer_agent
        self._prefix = prefix
        self._buffers: dict[int, list[torch.Tensor]] = {}
        self._mr_names: dict[int, list[str]] = {}

    def register(
        self,
        version: int,
        named_tensors: dict[str, torch.Tensor],
    ) -> WeightManifest:
        if version in self._buffers:
            raise RuntimeError(f"weight version {version} is already registered")

        bufs: list[torch.Tensor] = []
        mr_names: list[str] = []
        entries: list[TensorMRInfo] = []
        for name, t in named_tensors.items():
            cpu = (
                t.detach().to("cpu", non_blocking=False).contiguous()
                if not (t.device.type == "cpu" and t.is_contiguous())
                else t.detach().contiguous()
            )
            bufs.append(cpu)
            mr_name = f"{self._prefix}:{version}:{name}"
            size = cpu.numel() * cpu.element_size()
            self._agent.register_memory_region(mr_name, cpu.data_ptr(), 0, size)
            mr_names.append(mr_name)
            entries.append(
                TensorMRInfo(
                    name=name,
                    mr_name=mr_name,
                    size=size,
                    shape=tuple(cpu.shape),
                    dtype=_DTYPE_TO_STR[cpu.dtype],
                )
            )
        self._buffers[version] = bufs
        self._mr_names[version] = mr_names
        manifest = WeightManifest(
            version=version,
            train_alias=self._agent.alias,
            entries=entries,
        )
        logger.info(
            "WeightTransportTrain.register version=%d count=%d total_bytes=%d",
            version,
            len(entries),
            sum(e.size for e in entries),
        )
        return manifest

    def unregister(self, version: int) -> None:
        names = self._mr_names.pop(version, None)
        bufs = self._buffers.pop(version, None)
        if names is None:
            return
        for name in names:
            try:
                self._agent.unregister_memory_region(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("unregister_memory_region(%s) failed: %s", name, exc)
        del bufs
        logger.info("WeightTransportTrain.unregister version=%d", version)


# ---------------------------------------------------------------------------
# Rollout side
# ---------------------------------------------------------------------------


class WeightTransportRollout:
    """Pulls a ``WeightManifest`` over RDMA into local CPU receive buffers.

    Constructs the numeric ``[(local_handle, remote_handle, 0, 0, size)]``
    assign list once per pull (handles are resolved with one round-trip to
    Redis per remote MR) and issues a single batched
    ``endpoint.read(...)`` so the C++ side dispatches them as a unit.
    """

    def __init__(self, peer_agent: PeerAgent, train_alias: str):
        self._agent = peer_agent
        self._train_alias = train_alias
        self._receive: dict[int, dict[str, torch.Tensor]] = {}

    def pull(self, manifest: WeightManifest) -> dict[str, torch.Tensor]:
        # Get the underlying RDMAEndpoint for the train peer.
        endpoint = self._agent._get_endpoint(self._train_alias)
        conn = self._agent._get_connection(self._train_alias)

        receive_bufs: dict[str, torch.Tensor] = {}
        assigns: list[tuple[int, int, int, int, int]] = []
        for entry in manifest.entries:
            dtype = _STR_TO_DTYPE[entry.dtype]
            buf = torch.empty(entry.shape, dtype=dtype, device="cpu").contiguous()
            receive_bufs[entry.name] = buf
            self._agent.register_memory_region(
                entry.mr_name, buf.data_ptr(), 0, entry.size
            )
            local_handle = self._agent.get_handle(
                entry.mr_name,
                resource_key=conn.local_key,
            )
            remote_handle = self._agent.get_handle(
                entry.mr_name,
                self._train_alias,
                resource_key=conn.peer_key,
                endpoint=endpoint,
            )
            # Endpoint expects (local_handle, remote_handle, remote_offset,
            # local_offset, length). See dlslime/peer_agent/_agent.py:1487.
            assigns.append((local_handle, remote_handle, 0, 0, entry.size))

        slot = endpoint.read(assigns, None)
        slot.wait()

        self._receive[manifest.version] = receive_bufs
        logger.info(
            "WeightTransportRollout.pull version=%d count=%d total_bytes=%d",
            manifest.version,
            len(manifest.entries),
            sum(e.size for e in manifest.entries),
        )
        return receive_bufs

    def release(self, manifest: WeightManifest) -> None:
        """Release the receive buffers for ``manifest.version`` and
        unregister the MRs we registered on receive. Idempotent."""
        bufs = self._receive.pop(manifest.version, None)
        if bufs is None:
            return
        for entry in manifest.entries:
            try:
                self._agent.unregister_memory_region(entry.mr_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "unregister_memory_region(%s) failed: %s", entry.mr_name, exc
                )
        del bufs

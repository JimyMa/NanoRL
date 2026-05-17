"""Pytest fixtures shared across NanoRL tests."""

from __future__ import annotations

import os
import socket
import urllib.request

import pytest

DEFAULT_NANOCTRL_URL = os.environ.get("NANORL_NANOCTRL_URL", "http://127.0.0.1:3000")


def _nanoctrl_reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def nanoctrl_url() -> str:
    """NanoCtrl HTTP URL or skip the test."""
    url = DEFAULT_NANOCTRL_URL
    if not _nanoctrl_reachable(url):
        pytest.skip(
            f"NanoCtrl not reachable at {url}; set NANORL_NANOCTRL_URL or start it"
        )
    return url


@pytest.fixture(scope="session")
def has_rdma_device() -> bool:
    return os.path.isdir("/sys/class/infiniband") and any(
        os.listdir("/sys/class/infiniband")
    )


@pytest.fixture
def host_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

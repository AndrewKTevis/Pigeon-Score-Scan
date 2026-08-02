from __future__ import annotations

"""Block non-local socket connections in the published offline runtime."""

import ipaddress
import os
import sys
from typing import Any


def _is_loopback_address(address: Any) -> bool:
    if isinstance(address, str):
        return address.casefold() == "localhost"
    if not isinstance(address, tuple) or not address:
        return False
    host = str(address[0]).strip("[]")
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _offline_socket_guard(event: str, arguments: tuple[Any, ...]) -> None:
    if event != "socket.connect" or len(arguments) < 2:
        return
    if not _is_loopback_address(arguments[1]):
        raise PermissionError("Pigeon Score Scan offline runtime blocked a network connection")


if os.environ.get("SCORESCAN_OFFLINE_RUNTIME") == "1":
    sys.addaudithook(_offline_socket_guard)

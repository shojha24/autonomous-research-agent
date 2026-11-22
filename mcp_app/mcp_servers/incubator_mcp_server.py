"""MCP server exposing incubator operations as tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.incubator_outside import IncubatorCytomat


mcp = FastMCP("incubator")
_incubator: IncubatorCytomat | None = None


def _get_incubator(port: str) -> IncubatorCytomat:
    """Return a shared incubator instance for the provided serial port.

    Args:
        port: Serial port for connecting to the incubator hardware.

    Returns:
        A lazily initialized `IncubatorCytomat` object.
    """
    global _incubator
    if _incubator is None:
        _incubator = IncubatorCytomat(port=port)
    return _incubator


@mcp.tool()
def initialize(port: str) -> dict[str, Any]:
    """Initialize incubator connection.

    Args:
        port: Serial port path (for example, COM1 or /dev/ttyUSB0).

    Returns:
        Status payload indicating successful connection.
    """
    _get_incubator(port=port)
    return {"ok": True, "message": f"Connected to incubator on {port}."}


@mcp.tool()
def import_plate(incu_rack: int, incu_slot: int, port: str) -> dict[str, Any]:
    """Import a plate from transfer station to incubator slot.

    Args:
        incu_rack: Rack number in incubator.
        incu_slot: Slot number in selected rack.
        port: Serial port used for incubator connection.

    Returns:
        Status payload for operation result.
    """
    incubator = _get_incubator(port=port)
    incubator.import_plate(incu_rack=incu_rack, incu_slot=incu_slot)
    return {"ok": True, "message": f"Imported plate to rack {incu_rack}, slot {incu_slot}."}


@mcp.tool()
def export_plate(incu_rack: int, incu_slot: int, port: str) -> dict[str, Any]:
    """Export a plate from incubator slot to transfer station.

    Args:
        incu_rack: Rack number in incubator.
        incu_slot: Slot number in selected rack.
        port: Serial port used for incubator connection.

    Returns:
        Status payload for operation result.
    """
    incubator = _get_incubator(port=port)
    incubator.export_plate(incu_rack=incu_rack, incu_slot=incu_slot)
    return {"ok": True, "message": f"Exported plate from rack {incu_rack}, slot {incu_slot}."}


@mcp.tool()
def is_ready(port: str) -> dict[str, Any]:
    """Return incubator readiness status.

    Args:
        port: Serial port used for incubator connection.

    Returns:
        Payload containing readiness boolean.
    """
    incubator = _get_incubator(port=port)
    return {"ok": True, "ready": incubator.is_ready()}


if __name__ == "__main__":
    mcp.run(transport="stdio")

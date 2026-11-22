"""MCP server: exposes microscope stage/camera commands as MCP tools.

Tool names are prefixed with ``microscope_`` so they do not collide with
incubator tools (e.g. ``initialize``) when both servers are loaded by the host.

One shared Microscope connection per server process. First successful
``microscope_initialize`` opens hardware.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_servers.microscope_hardware import Microscope, RESULTS_DIR

mcp = FastMCP("microscope")

# One microscope instance for the whole MCP server process.
_microscope: Microscope | None = None


def _require_microscope() -> Microscope:
    """Return the shared microscope, or raise if not initialized.

    Returns:
        Connected Microscope instance.

    Raises:
        RuntimeError: If microscope_initialize has not been called yet.
    """
    if _microscope is None:
        raise RuntimeError(
            "Microscope not initialized. Call microscope_initialize first."
        )
    return _microscope


@mcp.tool()
def microscope_initialize(
    kcube_serial: str = "26004971",
    camera_serial: str = "25267",
    xy_port: str = "COM3",
    xy_baudrate: int = 9600,
) -> dict[str, Any]:
    """Connect to the microscope (Z-stage, camera, and XY stage).

    Args:
        kcube_serial: Thorlabs K-Cube serial number for the Z motor.
        camera_serial: Thorlabs TLCamera serial number.
        xy_port: Serial port for the XY stage (e.g. COM3 or /dev/ttyUSB1).
        xy_baudrate: XY stage baud rate.

    Returns:
        Dict with ok=True and a connection message on success, or ok=False
        with an error string on failure.
    """
    global _microscope
    if _microscope is not None:
        return {
            "ok": True,
            "message": "Microscope already connected; reusing existing session.",
        }
    try:
        _microscope = Microscope(
            kcube_serial=kcube_serial,
            camera_serial=camera_serial,
            xy_port=xy_port,
            xy_baudrate=xy_baudrate,
        )
    except Exception as exc:  # noqa: BLE001
        _microscope = None
        return {"ok": False, "error": f"Failed to connect microscope: {exc}"}
    return {
        "ok": True,
        "message": (
            f"Connected microscope (kcube={kcube_serial}, camera={camera_serial}, "
            f"xy_port={xy_port})."
        ),
        "results_dir": str(RESULTS_DIR),
    }


@mcp.tool()
def microscope_get_status() -> dict[str, Any]:
    """Return current X/Y/Z positions.

    Returns:
        Dict with ok, x (steps), y (steps), and z (meters).
    """
    try:
        scope = _require_microscope()
        status = scope.status()
        return {"ok": True, **status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_goto_xy(x: int, y: int) -> dict[str, Any]:
    """Move the XY stage to an absolute position (step units).

    Args:
        x: Absolute X step count.
        y: Absolute Y step count.

    Returns:
        Dict with ok and a status message.
    """
    try:
        scope = _require_microscope()
        ok = scope.goto_xy(x, y)
        if ok:
            return {"ok": True, "message": f"Moved XY to ({x}, {y}).", "x": x, "y": y}
        return {
            "ok": False,
            "error": f"Unable to move to ({x}, {y}); check limit switch / stage.",
            **scope.status(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_move_x(steps: int) -> dict[str, Any]:
    """Relative X move. Positive is right, negative is left.

    Args:
        steps: Signed step count.

    Returns:
        Dict with ok and updated status fields when possible.
    """
    try:
        scope = _require_microscope()
        ok = scope.move_x(steps)
        status = scope.status()
        if ok:
            return {"ok": True, "message": f"Moved X by {steps} steps.", **status}
        return {"ok": False, "error": f"Unable to move X by {steps}.", **status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_move_y(steps: int) -> dict[str, Any]:
    """Relative Y move. Positive is up, negative is down.

    Args:
        steps: Signed step count.

    Returns:
        Dict with ok and updated status fields when possible.
    """
    try:
        scope = _require_microscope()
        ok = scope.move_y(steps)
        status = scope.status()
        if ok:
            return {"ok": True, "message": f"Moved Y by {steps} steps.", **status}
        return {"ok": False, "error": f"Unable to move Y by {steps}.", **status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_move_z(meters: float) -> dict[str, Any]:
    """Move Z to an absolute position in meters.

    Args:
        meters: Absolute Z height in meters.

    Returns:
        Dict with ok and current z.
    """
    try:
        scope = _require_microscope()
        scope.move_z(meters)
        return {
            "ok": True,
            "message": f"Moved Z to {meters} m.",
            "z": scope.status()["z"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_move_z_relative(delta_meters: float) -> dict[str, Any]:
    """Move Z by a relative delta in meters.

    Args:
        delta_meters: Relative Z displacement in meters.

    Returns:
        Dict with ok and current z.
    """
    try:
        scope = _require_microscope()
        scope.move_z_relative(delta_meters)
        return {
            "ok": True,
            "message": f"Moved Z by {delta_meters} m.",
            "z": scope.status()["z"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_home_x() -> dict[str, Any]:
    """Home the X axis to 0.

    Returns:
        Dict with ok and status.
    """
    try:
        scope = _require_microscope()
        ok = scope.home_x()
        status = scope.status()
        if ok:
            return {"ok": True, "message": "Homed X axis.", **status}
        return {"ok": False, "error": "Could not home X axis.", **status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_home_y() -> dict[str, Any]:
    """Home the Y axis to 0.

    Returns:
        Dict with ok and status.
    """
    try:
        scope = _require_microscope()
        ok = scope.home_y()
        status = scope.status()
        if ok:
            return {"ok": True, "message": "Homed Y axis.", **status}
        return {"ok": False, "error": "Could not home Y axis.", **status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_home_z() -> dict[str, Any]:
    """Home the Z axis to 0 meters.

    Returns:
        Dict with ok and status.
    """
    try:
        scope = _require_microscope()
        scope.home_z()
        return {"ok": True, "message": "Homed Z axis.", **scope.status()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_home_all() -> dict[str, Any]:
    """Home X, Y, and Z.

    Returns:
        Dict with ok and status.
    """
    try:
        scope = _require_microscope()
        scope.home_all()
        return {"ok": True, "message": "Homed X, Y, and Z.", **scope.status()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_snap() -> dict[str, Any]:
    """Capture and save one image at the current stage position.

    Returns:
        Dict with ok, image_path, and current z when successful.
    """
    try:
        scope = _require_microscope()
        path = scope.snap()
        return {
            "ok": True,
            "message": "Image captured.",
            "image_path": str(path),
            "z": scope.status()["z"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_autofocus(
    z_start: float = 0.003,
    z_end: float = 0.0006,
    num_steps: int = 15,
    exposure: float = 0.01,
) -> dict[str, Any]:
    """Run a Tenengrad autofocus scan and move to the estimated best Z.

    This can take a long time (many Z steps + camera frames). Prefer human
    confirmation in the GUI before calling it on real hardware.

    Args:
        z_start: Start of Z scan range in meters.
        z_end: End of Z scan range in meters.
        num_steps: Number of sample positions.
        exposure: Camera exposure in seconds.

    Returns:
        Dict with ok, optimal_z, and optional plot_path.
    """
    try:
        scope = _require_microscope()
        optimal_z = scope.autofocus(
            z_start=z_start,
            z_end=z_end,
            num_steps=num_steps,
            exposure=exposure,
        )
        plot_path = RESULTS_DIR / "hill_climb.png"
        out: dict[str, Any] = {
            "ok": True,
            "message": "Autofocus complete.",
            "optimal_z": optimal_z,
            **scope.status(),
        }
        if plot_path.is_file():
            out["plot_path"] = str(plot_path)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def microscope_close() -> dict[str, Any]:
    """Close microscope hardware connections and clear the shared instance.

    Returns:
        Dict with ok and a status message.
    """
    global _microscope
    if _microscope is None:
        return {"ok": True, "message": "Microscope was not connected."}
    try:
        _microscope.close()
    except Exception as exc:  # noqa: BLE001
        _microscope = None
        return {"ok": False, "error": f"Error while closing: {exc}"}
    _microscope = None
    return {"ok": True, "message": "Microscope connections closed."}


if __name__ == "__main__":
    mcp.run(transport="stdio")

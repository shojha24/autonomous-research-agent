"""Human CLI for manual microscope testing (no LLM / MCP).

Uses the same Microscope driver as the MCP server. Run this at the bench when
you want to type commands directly:

    python -m mcp_servers.microscope_cli
    python -m mcp_servers.microscope_cli --xy-port COM3
"""

from __future__ import annotations

import argparse
import traceback

from pylablib.devices import Thorlabs

from mcp_servers.microscope_hardware import Microscope

HELP_TEXT = """
Available commands:
  mx <steps>          move X relative (+right / -left)
  my <steps>          move Y relative (+up / -down)
  goto <x> <y>        move X/Y to an absolute position
  mz <meters>         move Z to an absolute position
  dz <meters>         move Z relative to current position
  homex / homey / homez / home   home one axis or all axes
  autofocus [z_start z_end num_steps exposure]   run the autofocus scan
  snap                capture and save an image
  pos                 print current X/Y/Z position
  help                show this message
  quit                close connections and exit
"""


def run_cli(scope: Microscope) -> None:
    """Interactive command loop for manual microscope control.

    Args:
        scope: Connected Microscope instance.

    Returns:
        None.
    """
    print(HELP_TEXT)
    while True:
        try:
            raw = input("microscope> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "help":
                print(HELP_TEXT)
            elif cmd == "pos":
                print(scope.status())
            elif cmd == "mx":
                scope.move_x(int(args[0]))
            elif cmd == "my":
                scope.move_y(int(args[0]))
            elif cmd == "goto":
                scope.goto_xy(int(args[0]), int(args[1]))
            elif cmd == "mz":
                scope.move_z(float(args[0]))
            elif cmd == "dz":
                scope.move_z_relative(float(args[0]))
            elif cmd == "homex":
                scope.home_x()
            elif cmd == "homey":
                scope.home_y()
            elif cmd == "homez":
                scope.home_z()
            elif cmd == "home":
                scope.home_all()
            elif cmd == "autofocus":
                z_start = float(args[0]) if len(args) >= 1 else 0.003
                z_end = float(args[1]) if len(args) >= 2 else 0.0006
                num_steps = int(args[2]) if len(args) >= 3 else 15
                exposure = float(args[3]) if len(args) >= 4 else 0.01
                optimal = scope.autofocus(
                    z_start=z_start,
                    z_end=z_end,
                    num_steps=num_steps,
                    exposure=exposure,
                )
                print(f"Optimal z-position: {optimal:.6f}")
            elif cmd == "snap":
                path = scope.snap()
                print(f"Saved image to: {path}")
            else:
                print(f"Unknown command: {cmd!r}. Type 'help' for a list of commands.")
        except (IndexError, ValueError):
            print("Invalid or missing arguments for that command. Type 'help' for usage.")
        except Exception as exc:  # noqa: BLE001
            print(f"Command failed: {exc!r}")


def parse_args() -> argparse.Namespace:
    """Parse CLI connection arguments.

    Returns:
        Parsed argparse namespace with serial and device IDs.
    """
    parser = argparse.ArgumentParser(
        description="Manual CLI for microscope XY/Z/camera (bypasses MCP/LLM)."
    )
    parser.add_argument("--kcube-serial", default="26004971")
    parser.add_argument("--camera-serial", default="25267")
    parser.add_argument("--xy-port", default="COM3")
    parser.add_argument("--xy-baudrate", type=int, default=9600)
    return parser.parse_args()


def main() -> None:
    """Connect to hardware and run the interactive CLI until quit.

    Returns:
        None.
    """
    args = parse_args()
    print("Available cameras:", Thorlabs.list_cameras_tlcam())

    scope = Microscope(
        kcube_serial=args.kcube_serial,
        camera_serial=args.camera_serial,
        xy_port=args.xy_port,
        xy_baudrate=args.xy_baudrate,
    )

    try:
        run_cli(scope)
    except Exception as exc:  # noqa: BLE001
        print(f"\nScript stopped due to: {exc!r}")
        traceback.print_exc()
    finally:
        scope.close()


if __name__ == "__main__":
    main()

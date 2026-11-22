"""Hardware driver for the lab microscope: XY stage, Z-stage, and camera.

Talks to:
  - Custom XY stage over serial (POSITION:/CURRENTPOSITION protocol)
  - Thorlabs K-Cube Z-stage via pylablib
  - Thorlabs TLCamera via pylablib

Used by mcp_servers/microscope_mcp_server.py. Prefer raising exceptions over
sys.exit so the MCP layer can return {ok: false, error: ...} to the host.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import serial
from pylablib.devices import Thorlabs
from scipy.optimize import curve_fit

# Default image/plot output directory (created on first save).
RESULTS_DIR: Path = Path(__file__).resolve().parent / "microscope_results"


def gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """Evaluate a 1-D Gaussian used to fit the autofocus score curve.

    Args:
        x: Sample positions (z heights).
        amplitude: Peak height of the Gaussian.
        mu: Mean (peak location) of the Gaussian.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Array of Gaussian values at each x.
    """
    return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma**2))


class ZStageAutofocus:
    """Thorlabs Z-stage + TLCamera with Tenengrad autofocus."""

    def __init__(
        self,
        kcube_serial: str = "26004971",
        camera_serial: str = "25267",
        *,
        home_on_connect: bool = True,
    ) -> None:
        """Open the K-Cube motor and Thorlabs camera.

        Args:
            kcube_serial: Thorlabs Kinesis serial for the Z motor.
            camera_serial: Thorlabs TLCamera serial number.
            home_on_connect: If True and Z is not at 0, move to home on connect.

        Raises:
            Exception: Propagated from pylablib if hardware is missing.
        """
        self.kcube_serial = kcube_serial
        self.camera_serial = camera_serial

        self.z_stage = Thorlabs.KinesisMotor(kcube_serial, scale="ZFS13B")
        self.z_stage.setup_velocity(max_velocity=0.002, acceleration=0.005)

        self.camera = Thorlabs.ThorlabsTLCamera(serial=camera_serial)
        self.camera.set_exposure(0.0275)
        self.camera.set_trigger_mode("int")

        if home_on_connect:
            pos = self.get_z_position()
            if pos != 0:
                self.z_stage.move_to(0)
                time.sleep(1.0)

    def _camera_status_light(self) -> dict[str, Any]:
        """Collect lightweight camera status fields for debugging.

        Returns:
            Dict of trigger mode and acquisition state (or error strings).
        """
        info: dict[str, Any] = {}
        try:
            info["trigger_mode"] = self.camera.get_trigger_mode()
        except Exception as exc:  # noqa: BLE001
            info["trigger_mode_error"] = repr(exc)
        try:
            info["acquisition_in_progress"] = self.camera.acquisition_in_progress()
        except Exception as exc:  # noqa: BLE001
            info["acquisition_state_error"] = repr(exc)
        return info

    def _start_acquisition_loop(self, buffer_frames: int = 120) -> None:
        """Configure and start a software-triggered acquisition buffer.

        Args:
            buffer_frames: Frame buffer size for the camera acquisition.

        Returns:
            None.
        """
        try:
            self.camera.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.camera.clear_acquisition()
        except Exception:  # noqa: BLE001
            pass

        self.camera.set_trigger_mode("int")
        self.camera.setup_acquisition(nframes=buffer_frames)
        self.camera.start_acquisition(frames_per_trigger=1, auto_start=False)

    def _stop_acquisition_loop(self) -> None:
        """Stop and clear the camera acquisition buffer.

        Returns:
            None.
        """
        try:
            self.camera.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.camera.clear_acquisition()
        except Exception:  # noqa: BLE001
            pass

    def _capture_frame_low_level(
        self,
        timeout: float = 20.0,
        max_retries: int = 10,
    ) -> np.ndarray:
        """Acquire one frame with software trigger and retries.

        Args:
            timeout: Seconds to wait for a frame.
            max_retries: How many times to restart acquisition on failure.

        Returns:
            Captured image as a numpy array.

        Raises:
            RuntimeError: If no frame is returned after retries.
            Exception: Last camera error if all attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                while self.camera.read_oldest_image() is not None:
                    pass

                self.camera.send_software_trigger()
                self.camera.wait_for_frame(
                    since="now",
                    nframes=1,
                    timeout=timeout,
                    error_on_stopped=True,
                )
                img = self.camera.read_newest_image()
                if img is None:
                    raise RuntimeError("No frame returned after successful wait_for_frame")
                return img
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_retries:
                    time.sleep(0.2)
                    self._start_acquisition_loop(buffer_frames=120)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Frame capture failed with no exception detail")

    def _save_preview_image(self, img: np.ndarray) -> np.ndarray:
        """Convert high-bit-depth frames to 8-bit for JPG export.

        Args:
            img: Raw camera frame.

        Returns:
            uint8 image suitable for cv2.imwrite.
        """
        if img.dtype == np.uint8:
            return img
        maxv = float(np.max(img)) if img.size else 1.0
        if maxv <= 0:
            maxv = 1.0
        return np.clip((img.astype(np.float32) / maxv) * 255.0, 0, 255).astype(np.uint8)

    def _save_debug_frame(self, img: np.ndarray, z_position: float) -> Path:
        """Save a preview JPG under RESULTS_DIR/test_imgs.

        Args:
            img: Raw camera frame.
            z_position: Z height at capture time (meters); used only for logging.

        Returns:
            Path to the written JPG file.
        """
        del z_position  # kept for call-site clarity / future metadata
        output_dir = RESULTS_DIR / "test_imgs"
        output_dir.mkdir(parents=True, exist_ok=True)

        preview = self._save_preview_image(img)
        output_path = output_dir / f"{time.time():.3f}.jpg"
        cv2.imwrite(str(output_path), preview)
        return output_path

    def move_z(self, z_position: float) -> None:
        """Move the Z-stage to an absolute position in meters and wait.

        Args:
            z_position: Absolute Z position in meters.

        Returns:
            None.
        """
        self.z_stage.move_to(z_position)
        self.z_stage.wait_move()

    def move_z_relative(self, delta: float) -> None:
        """Move the Z-stage by a relative delta in meters.

        Args:
            delta: Relative displacement in meters.

        Returns:
            None.
        """
        current = self.get_z_position()
        self.move_z(current + delta)

    def home_z(self) -> None:
        """Return the Z-stage to position 0.

        Returns:
            None.
        """
        self.move_z(0.0)

    def get_z_position(self) -> float:
        """Read the current Z-stage position in meters.

        Returns:
            Current Z position.
        """
        return float(self.z_stage.get_full_info()["position"])

    def capture_image(self) -> tuple[np.ndarray, Path]:
        """Snap one image at the current Z position and save a preview JPG.

        Returns:
            Tuple of (raw image array, saved JPG path).
        """
        self._start_acquisition_loop(buffer_frames=120)
        try:
            img = self._capture_frame_low_level(timeout=20.0, max_retries=3)
            path = self._save_debug_frame(img, self.get_z_position())
            return img, path
        finally:
            self._stop_acquisition_loop()

    def tenengrad_focus(self, z_position: float) -> float:
        """Move to z_position, capture, and compute a Tenengrad focus score.

        Args:
            z_position: Absolute Z position in meters.

        Returns:
            Focus score (higher is sharper). Returns 0.0 if no strong edges.
        """
        self.z_stage.move_to(z_position)
        self.z_stage.wait_move()
        time.sleep(0.15)

        img = self._capture_frame_low_level(timeout=20.0, max_retries=3)
        self._save_debug_frame(img, z_position)

        gray = img.astype(np.float32)
        height, width = gray.shape
        y0, y1 = int(0.2 * height), int(0.8 * height)
        x0, x1 = int(0.2 * width), int(0.8 * width)
        gray = gray[y0:y1, x0:x1]

        mean = float(gray.mean())
        if mean > 1e-6:
            gray = gray / mean

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        g2 = gx * gx + gy * gy

        threshold = float(np.percentile(g2, 75))
        strong = g2[g2 > threshold]
        return float(strong.mean()) if strong.size else 0.0

    def autofocus(
        self,
        z_start: float = 0.003,
        z_end: float = 0.0006,
        num_steps: int = 15,
        exposure: float = 0.01,
    ) -> float:
        """Scan Z, fit a Gaussian to focus scores, and move to the peak.

        Args:
            z_start: Start of the Z scan range (meters).
            z_end: End of the Z scan range (meters).
            num_steps: Number of sample positions in the scan.
            exposure: Camera exposure time in seconds.

        Returns:
            Estimated optimal Z position in meters.
        """
        self.camera.set_exposure(exposure)

        z_positions = np.linspace(z_start, z_end, num_steps)
        sobel_focus: list[float] = []

        self._start_acquisition_loop(buffer_frames=max(120, num_steps * 4))
        try:
            for z_pos in z_positions:
                sobel_focus.append(self.tenengrad_focus(float(z_pos)))

            # Optional plot for humans debugging autofocus quality.
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 6))
                plt.scatter(z_positions, sobel_focus, s=50)
                plt.plot(z_positions, sobel_focus, "b-", alpha=0.3)
                plt.xlabel("Z Position (m)")
                plt.ylabel("Tenengrad Score")
                plt.title("Autofocus Curve")
                plt.grid(True, alpha=0.3)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                plot_path = RESULTS_DIR / "hill_climb.png"
                plt.savefig(plot_path)
                plt.close()
            except Exception:  # noqa: BLE001
                # Plotting is best-effort; autofocus still succeeds without it.
                pass

            guess_amp = max(sobel_focus)
            guess_mean = float(z_positions[int(np.argmax(sobel_focus))])
            guess_std = (z_end - z_start) / 6.0

            try:
                params, _covariance = curve_fit(
                    gaussian,
                    z_positions,
                    np.asarray(sobel_focus, dtype=float),
                    p0=[guess_amp, guess_mean, guess_std],
                )
                optimal_z = float(params[1])
            except RuntimeError:
                optimal_z = guess_mean

            self.tenengrad_focus(optimal_z)
            return optimal_z
        finally:
            self._stop_acquisition_loop()

    def close(self) -> None:
        """Close camera and Z-stage connections.

        Returns:
            None.
        """
        self.camera.close()
        self.z_stage.close()


class XYStage:
    """Custom XY stage controlled over a serial text protocol."""

    def __init__(self, port: str = "COM3", baudrate: int = 9600) -> None:
        """Open the serial port and wait for the stage START handshake.

        Args:
            port: Serial device path (e.g. COM3 or /dev/ttyUSB1).
            baudrate: Serial baud rate (default 9600).

        Raises:
            serial.SerialException: If the port cannot be opened.
            TimeoutError: If CURRENTPOSITION is never reported after START.
        """
        self.__hit_limit_switch = False
        try:
            self.ser = serial.Serial(port, baudrate, timeout=5)
        except serial.SerialException as exc:
            raise serial.SerialException(
                f"Could not open XY stage serial port {port}: {exc}"
            ) from exc

        time.sleep(1.0)
        self.ser.write(b"START")
        time.sleep(15.0)

        deadline = time.time() + 30.0
        line = b""
        while time.time() < deadline:
            line = self.ser.readline().rstrip()
            if line.split(b":")[0] == b"CURRENTPOSITION":
                break
        else:
            self.ser.close()
            raise TimeoutError(
                f"XY stage on {port} did not report CURRENTPOSITION after START"
            )

        time.sleep(1.0)
        position = line.split(b":")[1]
        self.x_pos = int(position.split(b",")[0])
        self.y_pos = int(position.split(b",")[1])

    def move_to(self, x_pos: int, y_pos: int) -> bool:
        """Move to an absolute XY position.

        Args:
            x_pos: Absolute X step count.
            y_pos: Absolute Y step count.

        Returns:
            True if the reported position matches the request.
        """
        if self._move(x_pos, y_pos):
            self.x_pos = x_pos
            self.y_pos = y_pos
            return True
        return False

    def move_down(self, distance: int) -> bool:
        """Move Y negative by distance steps.

        Args:
            distance: Steps to move.

        Returns:
            True on success.
        """
        if self._move(self.x_pos, self.y_pos - distance):
            self.y_pos -= distance
            return True
        return False

    def move_up(self, distance: int) -> bool:
        """Move Y positive by distance steps.

        Args:
            distance: Steps to move.

        Returns:
            True on success.
        """
        if self._move(self.x_pos, self.y_pos + distance):
            self.y_pos += distance
            return True
        return False

    def move_left(self, distance: int) -> bool:
        """Move X negative by distance steps.

        Args:
            distance: Steps to move.

        Returns:
            True on success.
        """
        if self._move(self.x_pos - distance, self.y_pos):
            self.x_pos -= distance
            return True
        return False

    def move_right(self, distance: int) -> bool:
        """Move X positive by distance steps.

        Args:
            distance: Steps to move.

        Returns:
            True on success.
        """
        if self._move(self.x_pos + distance, self.y_pos):
            self.x_pos += distance
            return True
        return False

    def _move(self, x_pos: int, y_pos: int) -> bool:
        """Send POSITION:x,y and wait until CURRENTPOSITION matches or limit hit.

        Args:
            x_pos: Target X steps.
            y_pos: Target Y steps.

        Returns:
            True if target reached; False if a limit switch reset occurred.
        """
        position_string = f"POSITION:{x_pos},{y_pos}"
        self.ser.write(position_string.encode())
        while True:
            line = self.ser.readline().rstrip()
            if line.split(b":")[0] == b"CURRENTPOSITION":
                position = line.split(b":")[1]
                reported_x = int(position.split(b",")[0])
                reported_y = int(position.split(b",")[1])
                if reported_x == x_pos and reported_y == y_pos:
                    return True
            if line == b"Error! Limits have been reached. Resetting...":
                self.__hit_limit_switch = True
                time.sleep(10.0)
                self.reset()
                return False

    def hit_limit_switch(self) -> bool:
        """Return and clear the latched limit-switch flag.

        Returns:
            True if a limit was hit since the last call.
        """
        if self.__hit_limit_switch:
            self.__hit_limit_switch = False
            return True
        return False

    def reset(self) -> None:
        """Reset tracked XY position to (0, 0) after a limit event.

        Returns:
            None.
        """
        time.sleep(5.0)
        self.x_pos = 0
        self.y_pos = 0

    def home_x_axis(self) -> bool:
        """Home the X axis to 0, keeping Y unchanged.

        Returns:
            True on success.
        """
        return self.move_to(0, self.y_pos)

    def home_y_axis(self) -> bool:
        """Home the Y axis to 0, keeping X unchanged.

        Returns:
            True on success.
        """
        return self.move_to(self.x_pos, 0)

    def close(self) -> None:
        """Close the serial port.

        Returns:
            None.
        """
        self.ser.close()


class Microscope:
    """Facade combining XY stage, Z-stage, and camera behind one API."""

    def __init__(
        self,
        kcube_serial: str = "26004971",
        camera_serial: str = "25267",
        xy_port: str = "COM3",
        xy_baudrate: int = 9600,
        *,
        home_z_on_connect: bool = True,
    ) -> None:
        """Connect Z/camera first, then the XY stage.

        Args:
            kcube_serial: Thorlabs K-Cube serial number.
            camera_serial: Thorlabs camera serial number.
            xy_port: Serial port for the XY stage.
            xy_baudrate: Baud rate for the XY stage.
            home_z_on_connect: If True, home Z when not already at 0.
        """
        self.z = ZStageAutofocus(
            kcube_serial=kcube_serial,
            camera_serial=camera_serial,
            home_on_connect=home_z_on_connect,
        )
        self.xy = XYStage(port=xy_port, baudrate=xy_baudrate)

    def move_x(self, distance: int) -> bool:
        """Relative X move. Positive is right, negative is left.

        Args:
            distance: Signed step count.

        Returns:
            True on success.
        """
        if distance >= 0:
            return self.xy.move_right(distance)
        return self.xy.move_left(-distance)

    def move_y(self, distance: int) -> bool:
        """Relative Y move. Positive is up, negative is down.

        Args:
            distance: Signed step count.

        Returns:
            True on success.
        """
        if distance >= 0:
            return self.xy.move_up(distance)
        return self.xy.move_down(-distance)

    def goto_xy(self, x_pos: int, y_pos: int) -> bool:
        """Absolute XY move.

        Args:
            x_pos: Absolute X steps.
            y_pos: Absolute Y steps.

        Returns:
            True on success.
        """
        return self.xy.move_to(x_pos, y_pos)

    def move_z(self, z_position: float) -> None:
        """Absolute Z move in meters.

        Args:
            z_position: Absolute Z in meters.

        Returns:
            None.
        """
        self.z.move_z(z_position)

    def move_z_relative(self, delta: float) -> None:
        """Relative Z move in meters.

        Args:
            delta: Relative Z displacement in meters.

        Returns:
            None.
        """
        self.z.move_z_relative(delta)

    def home_x(self) -> bool:
        """Home X only.

        Returns:
            True on success.
        """
        return self.xy.home_x_axis()

    def home_y(self) -> bool:
        """Home Y only.

        Returns:
            True on success.
        """
        return self.xy.home_y_axis()

    def home_z(self) -> None:
        """Home Z to 0.

        Returns:
            None.
        """
        self.z.home_z()

    def home_all(self) -> None:
        """Home X, then Y, then Z.

        Returns:
            None.
        """
        self.home_x()
        self.home_y()
        self.home_z()

    def autofocus(
        self,
        z_start: float = 0.003,
        z_end: float = 0.0006,
        num_steps: int = 15,
        exposure: float = 0.01,
    ) -> float:
        """Run Tenengrad autofocus and return the optimal Z.

        Args:
            z_start: Scan start in meters.
            z_end: Scan end in meters.
            num_steps: Number of Z samples.
            exposure: Camera exposure in seconds.

        Returns:
            Optimal Z position in meters.
        """
        return self.z.autofocus(
            z_start=z_start,
            z_end=z_end,
            num_steps=num_steps,
            exposure=exposure,
        )

    def snap(self) -> Path:
        """Capture and save one image; return the JPG path.

        Returns:
            Path to the saved preview image.
        """
        _img, path = self.z.capture_image()
        return path

    def status(self) -> dict[str, float | int]:
        """Return current X/Y/Z positions.

        Returns:
            Dict with integer x/y steps and float z meters.
        """
        return {
            "x": self.xy.x_pos,
            "y": self.xy.y_pos,
            "z": self.z.get_z_position(),
        }

    def close(self) -> None:
        """Close Z/camera and XY connections.

        Returns:
            None.
        """
        self.z.close()
        self.xy.close()

"""Acoustic pre-filtering, ultrasonic anomaly detection, and laboratory noise simulation.

Implements the physical and digital pre-processing defense layers specified in
Questions 2 & 3:
  1. 8th-order Butterworth low-pass filter (LPF) cutting off ultrasonic frequencies
     to eliminate hardware intermodulation distortion (VRIFLE, MetaAttack, SWhisper).
  2. High-frequency spectral anomaly detector computing energy ratios to flag stealthy
     acoustic injections before audio touches the ASR encoder.
  3. Laboratory noise simulation models for manual benchtop testing (fume hoods,
     centrifuges, vortex mixers, ultrasonic injection, and varying SNR levels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AcousticFilterConfig:
    """Configuration parameters for the digital acoustic pre-filter.

    Attributes:
        sample_rate: Expected audio sampling rate in Hz (default: 16000 Hz).
        cutoff_hz: Low-pass filter cutoff frequency in Hz (default: 7500.0 Hz).
        filter_order: Order of the Butterworth low-pass filter (default: 8).
        ultrasonic_anomaly_threshold: High-frequency energy ratio threshold to flag
            anomalous ultrasonic acoustic injection (default: 0.05).
        min_snr_db_threshold: Minimum allowable Signal-to-Noise Ratio in dB for
            high-confidence transcription (default: 0.0 dB).
    """

    sample_rate: int = 16000
    cutoff_hz: float = 7500.0
    filter_order: int = 8
    ultrasonic_anomaly_threshold: float = 0.05
    min_snr_db_threshold: float = 0.0


@dataclass
class FilteredAudioResult:
    """Container holding the results of acoustic filtration and security analysis.

    Attributes:
        filtered_audio: 1D numpy array of filtered audio samples (float32).
        sample_rate: Audio sampling rate in Hz.
        duration_sec: Duration of the audio chunk in seconds.
        estimated_snr_db: Estimated Signal-to-Noise Ratio in dB.
        ultrasonic_energy_ratio: Ratio of spectral energy above the cutoff band to total energy.
        is_ultrasonic_anomaly: True if ultrasonic energy ratio exceeds security threshold.
        security_message: Detailed diagnostic regarding acoustic safety.
    """

    filtered_audio: np.ndarray
    sample_rate: int
    duration_sec: float
    estimated_snr_db: float
    ultrasonic_energy_ratio: float
    is_ultrasonic_anomaly: bool
    security_message: str


class AcousticPreFilter:
    """Applies digital Butterworth LPF and spectral anomaly screening to raw PCM streams."""

    def __init__(self, config: AcousticFilterConfig | None = None) -> None:
        """Initialize filter with specified configuration.

        Args:
            config: Filter configuration settings. Defaults to AcousticFilterConfig().
        """
        self.config = config or AcousticFilterConfig()
        self._sos = self._build_butterworth_sos()

    def _build_butterworth_sos(self) -> np.ndarray:
        """Construct Second-Order Sections (SOS) for the 8th-order Butterworth LPF.

        Returns:
            Numpy array representing the SOS filter parameters.
        """
        nyquist = 0.5 * self.config.sample_rate
        # Ensure cutoff is safely below Nyquist
        normalized_cutoff = min(self.config.cutoff_hz / nyquist, 0.95)
        sos = signal.butter(
            N=self.config.filter_order,
            Wn=normalized_cutoff,
            btype="lowpass",
            output="sos",
        )
        return sos

    def filter_audio(self, audio: np.ndarray) -> np.ndarray:
        """Filter raw audio with the 8th-order Butterworth LPF.

        Args:
            audio: 1D float numpy array with values typically normalized between [-1.0, 1.0].

        Returns:
            Filtered 1D float32 numpy array.
        """
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        # Apply digital filter forward and backward for zero-phase delay if enough samples,
        # or forward SOS filter for streaming chunks.
        filtered = signal.sosfilt(self._sos, audio.astype(np.float64))
        return filtered.astype(np.float32)

    def compute_spectral_distribution(
        self, audio: np.ndarray, sample_rate: int | None = None
    ) -> dict[str, float]:
        """Compute the power spectral density and high-frequency energy ratio.

        Args:
            audio: Audio samples array.
            sample_rate: Sampling frequency in Hz (defaults to config sample_rate).

        Returns:
            Dict containing total_energy, audible_energy, high_band_energy,
            and high_band_ratio.
        """
        sr = sample_rate or self.config.sample_rate
        if len(audio) < 16:
            return {
                "total_energy": 0.0,
                "audible_energy": 0.0,
                "high_band_energy": 0.0,
                "high_band_ratio": 0.0,
            }

        # Calculate FFT power spectrum
        fft_vals = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        power_spectrum = np.abs(fft_vals) ** 2

        total_power = float(np.sum(power_spectrum))
        if total_power <= 1e-12:
            return {
                "total_energy": 0.0,
                "audible_energy": 0.0,
                "high_band_energy": 0.0,
                "high_band_ratio": 0.0,
            }

        # Split between normal speech band (< cutoff_hz) and potential carrier/ultrasound band
        cutoff = self.config.cutoff_hz
        high_mask = freqs >= cutoff
        high_band_power = float(np.sum(power_spectrum[high_mask]))
        audible_power = float(np.sum(power_spectrum[~high_mask]))

        high_band_ratio = high_band_power / total_power
        return {
            "total_energy": total_power,
            "audible_energy": audible_power,
            "high_band_energy": high_band_power,
            "high_band_ratio": high_band_ratio,
        }

    def estimate_snr_db(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio (SNR) in dB from signal energy variance.

        Args:
            audio: Audio samples array.

        Returns:
            Estimated SNR in decibels (dB).
        """
        if len(audio) < 100:
            return 30.0

        # Frame-based energy estimator: frame with lowest energy approximates noise floor
        frame_size = min(256, len(audio) // 4)
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return 30.0

        frame_energies = [
            float(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ]

        sorted_energies = sorted(frame_energies)
        # 10th percentile as estimated noise floor
        noise_idx = max(0, int(0.10 * len(sorted_energies)))
        noise_floor = sorted_energies[noise_idx] + 1e-9

        # 90th percentile as estimated speech level
        speech_idx = min(len(sorted_energies) - 1, int(0.90 * len(sorted_energies)))
        signal_level = sorted_energies[speech_idx] + 1e-9

        snr_ratio = signal_level / noise_floor
        snr_db = 10.0 * math.log10(max(snr_ratio, 1e-3))
        return float(np.clip(snr_db, -20.0, 60.0))

    def process(self, audio: np.ndarray | bytes) -> FilteredAudioResult:
        """Execute end-to-end acoustic filtration and security anomaly check.

        Args:
            audio: Raw PCM byte buffer (16-bit signed PCM) or float32 numpy array.

        Returns:
            FilteredAudioResult object.
        """
        # Convert bytes to float array if necessary
        if isinstance(audio, (bytes, bytearray)):
            int16_arr = np.frombuffer(audio, dtype=np.int16)
            audio_arr = (int16_arr / 32768.0).astype(np.float32)
        else:
            audio_arr = np.asarray(audio, dtype=np.float32)

        sr = self.config.sample_rate
        duration_sec = len(audio_arr) / float(sr) if sr > 0 else 0.0

        # 1. Analyze spectrum before filtering to detect ultrasonic carrier attacks
        spectral_stats = self.compute_spectral_distribution(audio_arr, sr)
        high_band_ratio = spectral_stats["high_band_ratio"]

        is_ultrasonic = high_band_ratio > self.config.ultrasonic_anomaly_threshold
        if is_ultrasonic:
            msg = (
                f"ALERT: Ultrasonic/near-ultrasound anomaly detected! "
                f"High-frequency energy ratio={high_band_ratio:.4f} exceeds threshold="
                f"{self.config.ultrasonic_anomaly_threshold}. Potential VRIFLE/MetaAttack carrier."
            )
        else:
            msg = (
                f"OK: High-frequency energy ratio={high_band_ratio:.4f} within normal bounds "
                f"(threshold={self.config.ultrasonic_anomaly_threshold})."
            )

        # 2. Apply 8th-order Butterworth low-pass filter
        filtered_audio = self.filter_audio(audio_arr)

        # 3. Estimate SNR
        snr_db = self.estimate_snr_db(filtered_audio)

        return FilteredAudioResult(
            filtered_audio=filtered_audio,
            sample_rate=sr,
            duration_sec=duration_sec,
            estimated_snr_db=snr_db,
            ultrasonic_energy_ratio=high_band_ratio,
            is_ultrasonic_anomaly=is_ultrasonic,
            security_message=msg,
        )


class SyntheticNoiseGenerator:
    """Generates synthetic laboratory acoustic profiles for testing under realistic constraints."""

    @staticmethod
    def generate_fume_hood_noise(
        duration_sec: float, sample_rate: int = 16000
    ) -> np.ndarray:
        """Synthesize low-frequency broadband laminar flow blower noise (pink/brownian drone).

        Args:
            duration_sec: Duration in seconds.
            sample_rate: Audio sampling rate in Hz.

        Returns:
            1D float32 numpy array.
        """
        num_samples = int(duration_sec * sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        # Generate white noise and pass through low-pass shaping (fume hood blower resonates < 400Hz)
        white_noise = np.random.normal(0, 1.0, num_samples)
        b, a = signal.butter(2, 350.0 / (sample_rate / 2.0), btype="lowpass")
        blower_noise = signal.lfilter(b, a, white_noise)

        # Add 60Hz and 120Hz motor hum
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        hum = 0.25 * np.sin(2 * np.pi * 60.0 * t) + 0.15 * np.sin(2 * np.pi * 120.0 * t)
        combined = blower_noise + hum
        # Normalize
        norm = np.max(np.abs(combined)) + 1e-9
        return (combined / norm).astype(np.float32)

    @staticmethod
    def generate_centrifuge_whistle(
        duration_sec: float,
        sample_rate: int = 16000,
        fundamental_hz: float = 1800.0,
    ) -> np.ndarray:
        """Synthesize high-speed benchtop centrifuge whine with distinct rotor harmonics.

        Args:
            duration_sec: Duration in seconds.
            sample_rate: Audio sampling rate in Hz.
            fundamental_hz: Primary rotor whistling frequency in Hz.

        Returns:
            1D float32 numpy array.
        """
        num_samples = int(duration_sec * sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        # Primary harmonic + 2nd + 3rd harmonics + light broadband air turbulence
        whistle = (
            0.60 * np.sin(2 * np.pi * fundamental_hz * t)
            + 0.25 * np.sin(2 * np.pi * (fundamental_hz * 2.0) * t)
            + 0.15 * np.sin(2 * np.pi * (fundamental_hz * 3.0) * t)
        )
        air_swirl = 0.10 * np.random.normal(0, 1.0, num_samples)
        combined = whistle + air_swirl
        norm = np.max(np.abs(combined)) + 1e-9
        return (combined / norm).astype(np.float32)

    @staticmethod
    def generate_vortex_mixer_noise(
        duration_sec: float, sample_rate: int = 16000, pulse_rate_hz: float = 24.0
    ) -> np.ndarray:
        """Synthesize rhythmic orbital vortex mixer vibration noise.

        Args:
            duration_sec: Duration in seconds.
            sample_rate: Audio sampling rate in Hz.
            pulse_rate_hz: Vibration pulsation rate in Hz.

        Returns:
            1D float32 numpy array.
        """
        num_samples = int(duration_sec * sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        # Amplitude modulated noise
        modulation = 0.5 * (1.0 + np.sin(2 * np.pi * pulse_rate_hz * t))
        carrier = np.random.normal(0, 1.0, num_samples)
        b, a = signal.butter(3, 800.0 / (sample_rate / 2.0), btype="lowpass")
        filtered_carrier = signal.lfilter(b, a, carrier)

        vortex = filtered_carrier * modulation
        norm = np.max(np.abs(vortex)) + 1e-9
        return (vortex / norm).astype(np.float32)

    @staticmethod
    def generate_ultrasonic_injection(
        duration_sec: float,
        sample_rate: int = 48000,
        carrier_hz: float = 19000.0,
    ) -> np.ndarray:
        """Synthesize near-ultrasonic carrier tone (e.g. 19-22kHz) simulating acoustic attacks.

        Args:
            duration_sec: Duration in seconds.
            sample_rate: High sampling rate supporting ultrasonic frequencies (e.g. 48kHz).
            carrier_hz: Ultrasonic tone frequency.

        Returns:
            1D float32 numpy array with strong ultrasonic carrier.
        """
        num_samples = int(duration_sec * sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        carrier = np.sin(2 * np.pi * carrier_hz * t)
        return carrier.astype(np.float32)

    @staticmethod
    def inject_noise_at_snr(
        clean_speech: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float,
    ) -> np.ndarray:
        """Mix noise into speech to achieve an exact target Signal-to-Noise Ratio (SNR).

        Args:
            clean_speech: 1D array of clean speech.
            noise: 1D array of noise (will be repeated or truncated to match speech length).
            target_snr_db: Desired SNR in decibels (e.g., 30 dB for quiet, 5 dB for loud lab).

        Returns:
            Combined noisy speech array normalized within [-1.0, 1.0].
        """
        if len(clean_speech) == 0:
            return clean_speech

        n_samples = len(clean_speech)
        # Tile or slice noise to match length
        if len(noise) < n_samples:
            repeats = int(math.ceil(n_samples / float(len(noise))))
            noise = np.tile(noise, repeats)[:n_samples]
        else:
            noise = noise[:n_samples]

        speech_power = float(np.mean(clean_speech**2)) + 1e-12
        noise_power = float(np.mean(noise**2)) + 1e-12

        # target_snr_db = 10 * log10(speech_power / scaled_noise_power)
        # scaled_noise_power = speech_power / (10 ** (target_snr_db / 10))
        target_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
        scaling_factor = math.sqrt(target_noise_power / noise_power)

        noisy_speech = clean_speech + (noise * scaling_factor)
        peak = float(np.max(np.abs(noisy_speech)))
        if peak > 1.0:
            noisy_speech = noisy_speech / peak
        return noisy_speech.astype(np.float32)

"""Frame-aligned MFCC and temporal-modulation features.

Both streams use a 30 ms Hamming window, a 20 ms hop, and a 512-point FFT.
MFCCs follow the openSMILE setup described by Ntalampiras et al. (2019):
23 HTK mel bands from 0 to 4000 Hz, coefficients 1-12, log frame energy, and
first through third temporal derivatives (52 values per frame).

The temporal-modulation stream is a Python stand-in for the MATLAB Modulation
Toolbox. Sixteen fourth-order gammatone channels (50-4000 Hz) are demodulated
with the Hilbert envelope. Each envelope is analyzed with a 160 ms window
centered on the MFCC frame, because a 30 ms window cannot resolve modulation
frequencies below about 33 Hz. Energy in 0-8, 8-16, 16-24, and 24-32 Hz is
log-compressed, giving 64 values per frame. The concatenated vector has 116
dimensions.

Standardization is intentionally not applied here. Each cross-validation fold
fits its own scaler on training frames only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.signal import get_window, hilbert, lfilter

from src.dataset import Recording, VadConfig

N_FEATURES = 116


@dataclass(frozen=True)
class FeatureConfig:
    """Constants that define the cached feature representation."""

    sample_rate: int = 8000
    frame_length: int = 240
    hop_length: int = 160
    n_fft: int = 512
    preemphasis: float = 0.97
    n_mels: int = 23
    n_mfcc: int = 12
    delta_width: int = 2
    fmin_hz: float = 0.0
    fmax_hz: float = 4000.0
    n_gammatone: int = 16
    gammatone_fmin: float = 50.0
    gammatone_fmax: float = 4000.0
    gammatone_taps: int = 512
    mod_window: int = 1280
    mod_n_fft: int = 2048
    mod_edges_hz: tuple[float, ...] = (0.0, 8.0, 16.0, 24.0, 32.0)

    @property
    def n_mfcc_features(self) -> int:
        return (self.n_mfcc + 1) * 4

    @property
    def n_modulation_features(self) -> int:
        return self.n_gammatone * (len(self.mod_edges_hz) - 1)

    @property
    def n_features(self) -> int:
        return self.n_mfcc_features + self.n_modulation_features


def _jsonable(config: FeatureConfig | VadConfig) -> dict[str, object]:
    payload = asdict(config)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def pipeline_hash(feature_config: FeatureConfig, vad_config: VadConfig) -> str:
    """Short hash of every constant that changes the cached matrices."""

    payload = {
        "features": _jsonable(feature_config),
        "vad": _jsonable(vad_config),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def recording_token(recording: Recording) -> str:
    """Identity of a clip, including size and modification time."""

    stat = recording.path.stat()
    resolved = str(recording.path.resolve())
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def preemphasis(y: NDArray[np.floating], coefficient: float) -> NDArray[np.float64]:
    """First-order high-pass used by the openSMILE MFCC front end."""

    signal = np.asarray(y, dtype=np.float64).reshape(-1)
    if signal.size < 2 or coefficient == 0.0:
        return signal.copy()
    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasized


def regression_delta(features: NDArray[np.floating], width: int = 2) -> NDArray[np.float64]:
    """openSMILE-style regression delta.

    ``d[t] = sum_{n=1}^{N} n (c[t+n] - c[t-n]) / (2 sum_{n=1}^{N} n^2)``.
    Edges are held at the first and last frames.
    """

    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("delta expects a (frames, features) array")
    n_frames = data.shape[0]
    if n_frames == 0 or width <= 0:
        return data.copy()
    padded = np.pad(data, ((width, width), (0, 0)), mode="edge")
    combined = np.zeros_like(data)
    denominator = 0.0
    for step in range(1, width + 1):
        forward = padded[width + step : width + step + n_frames]
        backward = padded[width - step : width - step + n_frames]
        combined += step * (forward - backward)
        denominator += float(step * step)
    return combined / (2.0 * denominator)


def _erb(frequency: float) -> float:
    return 24.7 * (4.37e-3 * frequency + 1.0)


def erb_centers(n_channels: int, fmin: float, fmax: float) -> NDArray[np.float64]:
    """Glasberg-Moore ERB-spaced center frequencies in Hz."""

    low = 21.4 * np.log10(4.37e-3 * fmin + 1.0)
    high = 21.4 * np.log10(4.37e-3 * fmax + 1.0)
    erbs = np.linspace(low, high, n_channels)
    return (10.0 ** (erbs / 21.4) - 1.0) / 4.37e-3


def gammatone_impulse(center_hz: float, sample_rate: int, n_taps: int) -> NDArray[np.float64]:
    """Real fourth-order gammatone impulse response, L2-normalized."""

    times = np.arange(n_taps, dtype=np.float64) / float(sample_rate)
    bandwidth = 1.019 * _erb(center_hz)
    envelope = (times ** 3) * np.exp(-2.0 * np.pi * bandwidth * times)
    envelope[0] = 0.0
    impulse = envelope * np.cos(2.0 * np.pi * center_hz * times)
    scale = float(np.sqrt(np.sum(impulse ** 2)))
    if scale > 0.0:
        impulse /= scale
    return impulse


def _reflect_pad(values: NDArray[np.floating], left: int, right: int) -> NDArray[np.float64]:
    """Pad the last axis, falling back to edge padding for very short inputs."""

    data = np.asarray(values, dtype=np.float64)
    pad_width = [(0, 0)] * (data.ndim - 1) + [(left, right)]
    try:
        return np.pad(data, pad_width, mode="reflect")
    except ValueError:
        return np.pad(data, pad_width, mode="edge")


def _hz_to_mel(frequency: NDArray[np.floating] | float) -> NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: NDArray[np.floating] | float) -> NDArray[np.float64]:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def htk_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> NDArray[np.float64]:
    """HTK mel matrix with Slaney area normalization.

    This is the same layout as ``librosa.filters.mel(..., htk=True, norm='slaney')``.
    It is implemented here so feature extraction does not import librosa, whose
    Numba dependency is blocked on some Windows machines.
    """

    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    weights = np.zeros((n_mels, fft_frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (fft_frequencies - left) / max(center - left, 1e-12)
        falling = (right - fft_frequencies) / max(right - center, 1e-12)
        triangle = np.maximum(0.0, np.minimum(rising, falling))
        weights[index] = triangle * (2.0 / max(right - left, 1e-12))
    return weights


def frame_count(n_samples: int, frame_length: int, hop_length: int) -> int:
    if n_samples < frame_length:
        return 1
    return 1 + (n_samples - frame_length) // hop_length


class FeatureExtractor:
    """Extract concatenated MFCC and temporal-modulation sequences."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        if self.config.n_features != N_FEATURES:
            raise ValueError(
                f"FeatureConfig produces {self.config.n_features} dimensions, "
                f"expected {N_FEATURES}."
            )
        self._mfcc_window = get_window(
            "hamming", self.config.frame_length, fftbins=True
        ).astype(np.float64)
        self._mod_window = get_window(
            "hamming", self.config.mod_window, fftbins=True
        ).astype(np.float64)
        self._mel = htk_mel_filterbank(
            self.config.sample_rate,
            self.config.n_fft,
            self.config.n_mels,
            self.config.fmin_hz,
            self.config.fmax_hz,
        )
        centers = erb_centers(
            self.config.n_gammatone,
            self.config.gammatone_fmin,
            self.config.gammatone_fmax,
        )
        self._impulses = [
            gammatone_impulse(float(center), self.config.sample_rate, self.config.gammatone_taps)
            for center in centers
        ]
        frequencies = np.fft.rfftfreq(
            self.config.mod_n_fft, d=1.0 / self.config.sample_rate
        )
        edges = self.config.mod_edges_hz
        self._mod_masks = [
            (frequencies >= low) & (frequencies < high)
            for low, high in zip(edges[:-1], edges[1:])
        ]
        if any(not np.any(mask) for mask in self._mod_masks):
            raise ValueError("A modulation-frequency band contains no FFT bins.")

    def transform(self, y: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return one ``(frames, 116)`` matrix for a trimmed waveform."""

        signal = np.asarray(y, dtype=np.float64).reshape(-1)
        if signal.size < self.config.frame_length:
            padded = np.zeros(self.config.frame_length, dtype=np.float64)
            padded[: signal.size] = signal
            signal = padded
        emphasized = preemphasis(signal, self.config.preemphasis)
        n_frames = frame_count(
            emphasized.size, self.config.frame_length, self.config.hop_length
        )
        mfcc = self._mfcc(emphasized, n_frames)
        modulation = self._temporal_modulation(emphasized, n_frames)
        features = np.concatenate([mfcc, modulation], axis=1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape != (n_frames, N_FEATURES):
            raise RuntimeError(
                f"Expected features {(n_frames, N_FEATURES)}, got {features.shape}."
            )
        return features

    def _power_frames(
        self, y: NDArray[np.float64], n_frames: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Power spectrum ``(bins, frames)`` and log energy ``(frames,)``."""

        config = self.config
        n_bins = config.n_fft // 2 + 1
        power = np.empty((n_bins, n_frames), dtype=np.float64)
        energy = np.empty(n_frames, dtype=np.float64)
        for index in range(n_frames):
            start = index * config.hop_length
            frame = y[start : start + config.frame_length]
            if frame.size < config.frame_length:
                padded_frame = np.zeros(config.frame_length, dtype=np.float64)
                padded_frame[: frame.size] = frame
                frame = padded_frame
            windowed = frame * self._mfcc_window
            energy[index] = np.log(np.sum(windowed ** 2) + 1e-10)
            spectrum = np.fft.rfft(windowed, n=config.n_fft)
            power[:, index] = spectrum.real ** 2 + spectrum.imag ** 2
        return power, energy

    def _mfcc(self, y: NDArray[np.float64], n_frames: int) -> NDArray[np.float64]:
        power, energy = self._power_frames(y, n_frames)
        mel = self._mel @ power
        log_mel = np.log(np.maximum(mel, 1e-10))
        cepstra = dct(log_mel, type=2, axis=0, norm="ortho")
        static = np.concatenate(
            [cepstra[1 : self.config.n_mfcc + 1].T, energy[:, None]],
            axis=1,
        )
        first = regression_delta(static, self.config.delta_width)
        second = regression_delta(first, self.config.delta_width)
        third = regression_delta(second, self.config.delta_width)
        return np.concatenate([static, first, second, third], axis=1)

    def _temporal_modulation(
        self, y: NDArray[np.float64], n_frames: int
    ) -> NDArray[np.float64]:
        envelopes = np.empty((len(self._impulses), y.size), dtype=np.float64)
        for index, impulse in enumerate(self._impulses):
            band = lfilter(impulse, np.array([1.0]), y)
            envelopes[index] = np.abs(hilbert(band))

        half = self.config.mod_window // 2
        padded = _reflect_pad(envelopes, half, half)
        offset = self.config.frame_length // 2
        starts = offset + np.arange(n_frames) * self.config.hop_length
        needed = int(starts[-1] + self.config.mod_window) if n_frames else 0
        if padded.shape[-1] < needed:
            padded = _reflect_pad(padded, 0, needed - padded.shape[-1])
        columns = starts[:, None] + np.arange(self.config.mod_window)[None, :]
        windowed = padded[:, columns] * self._mod_window[None, None, :]
        magnitude = np.abs(np.fft.rfft(windowed, n=self.config.mod_n_fft, axis=-1))
        bands = [
            np.log(np.sum(magnitude[:, :, mask], axis=-1) + 1e-10)
            for mask in self._mod_masks
        ]
        stacked = np.stack(bands, axis=-1)
        return stacked.transpose(1, 0, 2).reshape(n_frames, -1)


def save_feature_cache(
    path: Path,
    sequences: list[NDArray[np.float64]],
    recordings: list[Recording],
    config_hash: str,
) -> None:
    """Write concatenated frame matrices and the metadata needed to reload them."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sequences:
        raise ValueError("Cannot cache an empty feature set.")
    features = np.concatenate(sequences, axis=0).astype(np.float64, copy=False)
    lengths = np.asarray([sequence.shape[0] for sequence in sequences], dtype=np.int32)
    np.savez_compressed(
        path,
        features=features,
        lengths=lengths,
        labels=np.asarray([recording.label for recording in recordings]),
        cat_ids=np.asarray([recording.cat_id for recording in recordings]),
        breeds=np.asarray([recording.breed for recording in recordings]),
        filenames=np.asarray([recording.filename for recording in recordings]),
        tokens=np.asarray([recording_token(recording) for recording in recordings]),
        config_hash=np.asarray(config_hash),
    )


def load_feature_cache(
    path: Path,
    recordings: list[Recording],
    config_hash: str,
) -> list[NDArray[np.float64]] | None:
    """Load cached sequences when the hash and file list still match."""

    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        cached_hash = str(data["config_hash"])
        tokens = [str(token) for token in data["tokens"]]
        expected = [recording_token(recording) for recording in recordings]
        if cached_hash != config_hash or tokens != expected:
            return None
        features = np.asarray(data["features"], dtype=np.float64)
        lengths = np.asarray(data["lengths"], dtype=np.int32)
    sequences: list[NDArray[np.float64]] = []
    cursor = 0
    for length in lengths:
        end = cursor + int(length)
        sequences.append(features[cursor:end].copy())
        cursor = end
    if cursor != len(features):
        return None
    return sequences

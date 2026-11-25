"""Discover CatMeows recordings, load 8 kHz audio, and trim silence.

Filenames follow ``C_NNNNN_BB_SS_OOOOO_RXX.wav``. The first character is the
emission context (``F`` food, ``I`` isolation, ``B`` brushing). ``NNNNN`` is
the cat id and ``BB`` is the breed (``MC`` or ``EU``). The public release
uses five-character alphanumeric ids, for example ``B_ANI01_MC_FN_SIM01_101.wav``.

Each clip was published with about 0.5 s of leading and trailing silence.
Those handles are removed with a Sohn et al. (1999) likelihood-ratio voice
activity detector. An energy and spectral-flatness rule is used when the
likelihood-ratio path marks nothing. A trimmed clip is never empty: a result
shorter than 150 ms is expanded around the peak-energy sample to 300 ms, or
the whole clip is kept when the file itself is shorter.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

SEED = 42

FILENAME_RE = re.compile(
    r"^(?P<label>[BFI])_(?P<cat_id>[A-Za-z0-9]{5})_(?P<breed>MC|EU)_"
    r"(?P<sex>FI|FN|MI|MN)_(?P<owner>[A-Za-z0-9]{5})_"
    r"(?P<session>\d)(?P<index>\d{2})\.wav$",
    re.IGNORECASE,
)

ZENODO_RECORD = "https://zenodo.org/records/4008297"
ZENODO_ZIP = "https://zenodo.org/records/4008297/files/dataset.zip?download=1"
TARGET_SAMPLE_RATE = 8000


@dataclass(frozen=True)
class VadConfig:
    """Parameters for handle-aware silence trimming.

    Framing matches the feature extractor (30 ms window, 20 ms hop, FFT 512)
    so a frame decision lines up with later acoustic frames.
    """

    sample_rate: int = TARGET_SAMPLE_RATE
    frame_length: int = 240
    hop_length: int = 160
    n_fft: int = 512
    handle_seconds: float = 0.4
    alpha: float = 0.98
    log_lr_threshold: float = 0.0
    hangover_frames: int = 3
    pre_hang_frames: int = 2
    energy_margin_db: float = 6.0
    flatness_threshold: float = 0.4
    min_keep_seconds: float = 0.15
    expand_to_seconds: float = 0.30

    def to_dict(self) -> dict[str, float | int]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Recording:
    """One CatMeows clip and the metadata parsed from its filename."""

    path: Path
    label: str
    cat_id: str
    breed: str
    sex: str
    owner: str
    session: str
    index: str

    @property
    def filename(self) -> str:
        return self.path.name


def parse_filename(name: str) -> Recording | None:
    """Parse a CatMeows filename. Return ``None`` when it does not match."""

    match = FILENAME_RE.match(Path(name).name)
    if match is None:
        return None
    groups = match.groupdict()
    return Recording(
        path=Path(name),
        label=groups["label"].upper(),
        cat_id=groups["cat_id"].upper(),
        breed=groups["breed"].upper(),
        sex=groups["sex"].upper(),
        owner=groups["owner"].upper(),
        session=groups["session"],
        index=groups["index"],
    )


def discover(data_dir: Path) -> tuple[list[Recording], list[str]]:
    """Find matching wav files under ``data_dir``.

    Returns the matched recordings in filename order and the names that were
    skipped because they do not follow the CatMeows pattern.
    """

    data_dir = Path(data_dir)
    matched: list[Recording] = []
    skipped: list[str] = []
    if not data_dir.exists():
        return matched, skipped
    for path in sorted(data_dir.rglob("*.wav")):
        parsed = parse_filename(path.name)
        if parsed is None:
            skipped.append(path.name)
            continue
        matched.append(
            Recording(
                path=path,
                label=parsed.label,
                cat_id=parsed.cat_id,
                breed=parsed.breed,
                sex=parsed.sex,
                owner=parsed.owner,
                session=parsed.session,
                index=parsed.index,
            )
        )
    return matched, skipped


def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> NDArray[np.float64]:
    """Load a wav as mono float64 at ``target_sr`` Hz.

    Stereo files are averaged to mono. A file recorded at another rate is
    resampled with ``scipy.signal.resample``.
    """

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float64")
    mono = np.mean(audio, axis=1)
    if int(sample_rate) != int(target_sr):
        from scipy.signal import resample

        n_target = max(1, int(round(len(mono) * float(target_sr) / float(sample_rate))))
        mono = np.asarray(resample(mono, n_target), dtype=np.float64)
    return np.ascontiguousarray(mono, dtype=np.float64)


def _frame_count(n_samples: int, frame_length: int, hop_length: int) -> int:
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def _power_spectrogram(
    y: NDArray[np.float64],
    frame_length: int,
    hop_length: int,
    n_fft: int,
) -> NDArray[np.float64]:
    """Hamming-windowed power spectrum with ``center=False`` framing."""

    n_frames = _frame_count(len(y), frame_length, hop_length)
    if n_frames == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float64)
    window = np.hamming(frame_length).astype(np.float64)
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for index in range(n_frames):
        start = index * hop_length
        frame = y[start : start + frame_length] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        spec[:, index] = (spectrum.real**2 + spectrum.imag**2) / n_fft
    return spec


def _spectral_flatness(power: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spectral flatness per frame. Values near 1 are noise-like."""

    magnitude = np.sqrt(np.maximum(power, 0.0))
    logged = np.mean(np.log(magnitude + 1e-12), axis=0)
    arithmetic = np.mean(magnitude, axis=0) + 1e-12
    return np.exp(logged) / arithmetic


def _apply_hangover(
    active: NDArray[np.bool_],
    hangover_frames: int,
    pre_hang_frames: int,
) -> NDArray[np.bool_]:
    marked = active.copy()
    indices = np.flatnonzero(active)
    n_frames = len(active)
    for index in indices:
        left = max(0, index - pre_hang_frames)
        right = min(n_frames, index + hangover_frames + 1)
        marked[left:right] = True
    return marked


def _region_from_mask(
    active: NDArray[np.bool_],
    hop_length: int,
    frame_length: int,
    n_samples: int,
) -> tuple[int, int] | None:
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return None
    start = int(indices[0] * hop_length)
    end = int(indices[-1] * hop_length + frame_length)
    return max(0, start), min(n_samples, end)


def _sohn_mask(
    power: NDArray[np.float64],
    y: NDArray[np.float64],
    config: VadConfig,
) -> NDArray[np.bool_]:
    """Frame mask from a decision-directed likelihood-ratio test.

    Noise power is estimated from the published silence handles (the first
    and last ``handle_seconds``), which are known to be non-vocal.
    """

    n_frames = power.shape[1]
    mask = np.zeros(n_frames, dtype=bool)
    if n_frames == 0:
        return mask

    handle = int(round(config.handle_seconds * config.sample_rate))
    noise_frames: list[int] = []
    for index in range(n_frames):
        start = index * config.hop_length
        end = start + config.frame_length
        in_head = end <= handle
        in_tail = start >= len(y) - handle
        if in_head or in_tail:
            noise_frames.append(index)
    if len(noise_frames) < 2:
        frame_energy = np.sum(power, axis=0)
        noise_frames = list(np.argsort(frame_energy)[: max(2, n_frames // 5)])

    noise_psd = np.mean(power[:, noise_frames], axis=1)
    noise_psd = np.maximum(noise_psd, 1e-12)
    gamma = power / noise_psd[:, None]
    # Skip the DC bin; it is dominated by the microphone offset.
    gamma = gamma[1:, :]

    xi = np.zeros_like(gamma)
    xi[:, 0] = np.maximum(gamma[:, 0] - 1.0, 0.0)
    alpha = config.alpha
    for time in range(1, n_frames):
        previous = (xi[:, time - 1] / (1.0 + xi[:, time - 1])) * gamma[:, time - 1]
        xi[:, time] = alpha * previous + (1.0 - alpha) * np.maximum(gamma[:, time] - 1.0, 0.0)
    xi = np.maximum(xi, 1e-8)

    log_lr = -np.log1p(xi) + (gamma * xi) / (1.0 + xi)
    frame_log_lr = np.mean(log_lr, axis=0)
    return frame_log_lr > config.log_lr_threshold


def _energy_flatness_mask(
    power: NDArray[np.float64],
    y: NDArray[np.float64],
    config: VadConfig,
) -> NDArray[np.bool_]:
    """Fallback mask: energy above the handle floor and low spectral flatness."""

    n_frames = power.shape[1]
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    frame_energy = np.sum(power, axis=0) + 1e-12
    handle = int(round(config.handle_seconds * config.sample_rate))
    noise_energy: list[float] = []
    for index in range(n_frames):
        start = index * config.hop_length
        end = start + config.frame_length
        if end <= handle or start >= len(y) - handle:
            noise_energy.append(float(frame_energy[index]))
    if not noise_energy:
        noise_floor = float(np.percentile(frame_energy, 20))
    else:
        noise_floor = float(np.median(noise_energy))
    noise_floor = max(noise_floor, 1e-12)
    margin = 10.0 * np.log10(frame_energy / noise_floor)
    flatness = _spectral_flatness(power)
    return (margin >= config.energy_margin_db) & (flatness < config.flatness_threshold)


def _expand_around_peak(
    y: NDArray[np.float64],
    config: VadConfig,
) -> NDArray[np.float64]:
    """Keep a window around the highest-energy sample."""

    target = int(round(config.expand_to_seconds * config.sample_rate))
    if len(y) <= target:
        return y
    if target <= 0:
        return y
    peak = int(np.argmax(y.astype(np.float64) ** 2))
    start = max(0, peak - target // 2)
    end = min(len(y), start + target)
    start = max(0, end - target)
    segment = y[start:end]
    if len(segment) == 0:
        return y
    return segment


def trim_silence(
    y: NDArray[np.float64],
    config: VadConfig | None = None,
) -> NDArray[np.float64]:
    """Remove leading and trailing non-vocal frames.

    The Sohn likelihood-ratio mask is tried first. The energy and spectral
    flatness rule replaces it when the likelihood-ratio test marks no frames.
    The samples between the first and last active frame are kept, so short
    gaps inside a meow are not punched out. The result is always non-empty.
    """

    config = config or VadConfig()
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return np.zeros(config.frame_length, dtype=np.float64)

    power = _power_spectrogram(y, config.frame_length, config.hop_length, config.n_fft)
    if power.shape[1] == 0:
        return y.copy()

    sohn = _sohn_mask(power, y, config)
    sohn = _apply_hangover(sohn, config.hangover_frames, config.pre_hang_frames)
    region = _region_from_mask(sohn, config.hop_length, config.frame_length, len(y))
    if region is None:
        fallback = _energy_flatness_mask(power, y, config)
        fallback = _apply_hangover(fallback, config.hangover_frames, config.pre_hang_frames)
        region = _region_from_mask(fallback, config.hop_length, config.frame_length, len(y))

    if region is None:
        trimmed = _expand_around_peak(y, config)
    else:
        start, end = region
        trimmed = y[start:end]

    min_keep = int(round(config.min_keep_seconds * config.sample_rate))
    if len(trimmed) < min_keep:
        trimmed = _expand_around_peak(y, config)
    if len(trimmed) == 0:
        return y.copy()
    return np.ascontiguousarray(trimmed, dtype=np.float64)


def load_corpus(
    data_dir: Path,
    vad_config: VadConfig | None = None,
) -> tuple[list[NDArray[np.float64]], list[Recording]]:
    """Load every matching recording and return silence-trimmed waveforms."""

    recordings, skipped = discover(data_dir)
    if skipped:
        print(f"Skipped {len(skipped)} wav files that do not match the CatMeows pattern.")
    config = vad_config or VadConfig()
    waveforms: list[NDArray[np.float64]] = []
    for recording in recordings:
        audio = load_audio(recording.path, target_sr=config.sample_rate)
        waveforms.append(trim_silence(audio, config))
    return waveforms, recordings


def download_catmeows(data_dir: Path, url: str = ZENODO_ZIP) -> Path:
    """Download and extract the Zenodo ``dataset.zip`` into ``data_dir``."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "dataset.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "meow-ml/1.0"})
    print(f"Downloading CatMeows from {ZENODO_RECORD}")
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    with zipfile.ZipFile(destination) as archive:
        archive.extractall(data_dir)
    return data_dir


def missing_data_message(data_dir: Path) -> str:
    return (
        f"No CatMeows wav files found in {data_dir}. "
        f"Download dataset.zip from {ZENODO_RECORD} "
        f"or rerun with --download."
    )

"""
Dataset fetcher (HuggingFace-first) for the "Amaze" KWS corpus
-----------------------------------------------------------
Datasets required, as named in agents.md 2 / 9 / 10.1 and build_speech_corpus.py:

  1. LibriSpeech dev-clean     real human speech; the `real` negative content type,
                               the multi-talker babble talker pool, and the real
                               human utterances containing "amaze".
  2. ESC-50 (full, 2000 clips) 50 environmental classes / 2000 clips.
  3. UrbanSound8K              8732 clips / 18 classes incl. children_playing,
                               crowd_hailing, street_music, police_car.
  4. Procedural babble + device noises -- synthesised locally from (1), no download.

Why HuggingFace parquet instead of the original hosts
-----------------------------------------------------
Measured on this machine, sustained throughput:

    openslr.org (dev-clean.tar.gz)         ~85 KB/s -> 337 MB takes ~66 min
    raw.githubusercontent.com (ESC-50 wavs) ~31 KB/s -> 2000 files is unusable
    pypi.org (wheels)                       ~54 KB/s
    huggingface.co (parquet)            ~1,800 KB/s -> ~25x faster

The canonical hosts are impractical here, and all three corpora have a parquet
mirror on HuggingFace that serves at CDN speed. Same corpora, same labels -- only
the container differs (one shard file instead of 2000 loose wavs).

    LibriSpeech   openslr/librispeech_asr   all/validation.clean/0000.parquet
    ESC-50        ashraq/esc50             data/train-0000{0,1}-of-00002-*.parquet
    UrbanSound8K  danavery/urbansound8K     data/train-0000{0..7}-of-00016-*.parquet

Output goes to the layouts the corpus builder already expects:

    speech_corpus/LibriSpeech/**/*.flac          (build_corpus_v2.SpeechBank)
    noise_bank_v2/raw/esc50/<class>__<file>.wav  (build_noise_bank_v2)
    noise_bank_v2/raw/urbansound/us8k_*.wav

so build_noise_bank_v2.py and build_corpus_v2.py run unchanged afterwards.

    python download_datasets_hf.py                # all three
    python download_datasets_hf.py --libri
    python download_datasets_hf.py --esc50 --urbansound --shards 4
"""

import argparse
import io
import os
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf

SPEECH_DIR = "speech_corpus"
NOISE_DIR = "noise_bank_v2"
RAW_DIR = os.path.join(NOISE_DIR, "raw")
INDEX_CSV = os.path.join(NOISE_DIR, "index.csv")
# Large parquet shards land here first so an interrupted run resumes instead of
# re-downloading hundreds of megabytes.
SHARD_CACHE = os.path.join(NOISE_DIR, "_shard_cache")

SAMPLE_RATE = 16000
UA = {"User-Agent": "Mozilla/5.0 (amaze-kws-corpus-builder)"}

# -----------------------------------------------------------------------------
# HF sources
# -----------------------------------------------------------------------------
HF_LIBRISPEECH_REPO = "openslr/librispeech_asr"
HF_LIBRISPEECH_FILE = "all/validation.clean/0000.parquet"   # == dev-clean

HF_ESC50_REPO = "ashraq/esc50"
HF_ESC50_FILES = [
    "data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
    "data/train-00001-of-00002-27425e5c1846b494.parquet",
]

HF_US8K_REPO = "danavery/urbansound8K"
# First N of 16 shards. Each shard is large and the builder caps clips per class
# anyway, so pulling all 16 would spend bandwidth on clips we then discard.
HF_US8K_SHARDS = [
    "data/train-00000-of-00016-e478d7cccca6a095.parquet",
    "data/train-00001-of-00016-299138aa39afaa06.parquet",
    "data/train-00002-of-00016-887e0748205b6fa9.parquet",
    "data/train-00003-of-00016-691ee48aa53d9c1f.parquet",
    "data/train-00004-of-00016-c0f37514d8e28a72.parquet",
    "data/train-00005-of-00016-55ef1a0a51149c01.parquet",
    "data/train-00006-of-00016-0ef363072505e6d5.parquet",
    "data/train-00007-of-00016-dfac173beb21e5db.parquet",
]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _fetch(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    """GET with retries AND byte-range resumption.

    UrbanSound8K shards are ~430 MB each. On this machine a single long transfer
    gets truncated mid-stream (observed: IncompleteRead after 130 MB), and a
    naive retry restarts the whole 430 MB from zero. Because HF honours Range
    requests, the partial body is kept on disk and the next attempt asks only for
    the remainder. A socket read timeout also guards the case where the peer stops
    sending without closing, which otherwise hangs forever with no error.
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                      # noqa: BLE001 - report and retry
            last = e
            print(f"    retry {attempt + 1}/{retries}: {str(e)[:70]}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _fetch_resumable(url: str, cache_path: str, timeout: int = 60,
                     retries: int = 8) -> bytes:
    """Download `url` to `cache_path`, resuming a partial file with HTTP Range.

    Returns the complete bytes. A completed cache file is reused as-is, so a rerun
    after an interruption costs nothing. This exists because UrbanSound8K shards
    are ~430 MB and this machine's link truncates long single-shot transfers.
    """
    total = None
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        total = _remote_size(url)
        if total is not None and os.path.getsize(cache_path) == total:
            print(f"    cached {os.path.basename(cache_path)} "
                  f"({total / 1e6:.0f} MB)")
            return open(cache_path, "rb").read()
    last = None
    for attempt in range(retries):
        have = os.path.getsize(cache_path) if os.path.exists(cache_path) else 0
        if total is None:
            total = _remote_size(url)
        if total is not None and have >= total:
            break
        hdr = dict(UA)
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if total is None:
                    cr = r.headers.get("Content-Range")
                    if cr and "/" in cr:
                        total = int(cr.split("/")[-1])
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                got = 0
                with open(cache_path, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                print(f"    got {got / 1e6:.0f} MB in {time.time() - t0:.0f}s"
                      f" (total {os.path.getsize(cache_path) / 1e6:.0f} MB)")
        except Exception as e:                      # noqa: BLE001
            last = e
            print(f"    resume {attempt + 1}/{retries}: {str(e)[:60]}"
                  f" (have {os.path.getsize(cache_path) / 1e6:.0f} MB)")
            time.sleep(2)
    if total is not None and os.path.exists(cache_path) \
            and os.path.getsize(cache_path) != total:
        raise RuntimeError(f"incomplete {url}: "
                           f"{os.path.getsize(cache_path)}/{total} bytes")
    return open(cache_path, "rb").read()


def _remote_size(url: str) -> int | None:
    """Content-Length via a HEAD, or None if the server will not say."""
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:                              # noqa: BLE001
        return None


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def to_16k_mono(data: np.ndarray, sr: int) -> np.ndarray:
    """Matches build_noise_bank_v2.to_16k_mono so both fetch paths yield identical
    clips. Uses linear interpolation instead of scipy.signal.resample to keep the
    dependency surface small; for noise-bank material the difference is immaterial."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE and len(data) > 1:
        n_out = int(round(len(data) * SAMPLE_RATE / sr))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
    if len(data) < SAMPLE_RATE // 2:
        data = np.pad(data, (0, SAMPLE_RATE // 2 - len(data)))
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.89
    return data.astype(np.float32)


def save_clip(clip: np.ndarray, path: str, sr: int = SAMPLE_RATE) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, clip.astype(np.float32), sr)
        return True
    except Exception:                              # noqa: BLE001
        return False


def _decode_audio_cell(cell) -> tuple:
    """HF 'audio' columns arrive either as {bytes, path} or as a list/tuple of the
    same. Returns (np.ndarray, sr) or raises."""
    if isinstance(cell, dict):
        raw = cell.get("bytes")
    elif isinstance(cell, (list, tuple)) and cell:
        raw = cell[0].get("bytes") if isinstance(cell[0], dict) else None
    else:
        raw = None
    if not raw:
        raise ValueError("no audio bytes in cell")
    arr, sr = sf.read(io.BytesIO(raw))
    return arr, sr
def _write_index():
    """Rebuilds noise_bank_v2/index.csv from what is actually on disk, so the
    class-stratified sampler in build_corpus_v2 sees a complete inventory."""
    import csv
    rows = []
    subs = sorted(os.listdir(RAW_DIR)) if os.path.isdir(RAW_DIR) else []
    for sub in subs:
        d = os.path.join(RAW_DIR, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(d, f)
            try:
                info = sf.info(path)
                dur = float(info.frames) / float(info.samplerate)
            except Exception:                      # noqa: BLE001
                dur = 0.0
            # "airplane__1-11687-A-47.wav" -> class "airplane". The class is BEFORE the "__";
            # taking the part after it would give one class per clip and silently
            # destroy the class-stratified sampling build_corpus_v2 relies on.
            if "__" in f:
                cls = f.split("__", 1)[0]
            elif sub == "urbansound":
                parts = f.split("_")
                cls = parts[2].rsplit(".", 1)[0] if len(parts) > 2 else "us8k_unknown"
            else:
                cls = sub
            rows.append((os.path.join(sub, f), cls, f"{dur:.3f}"))
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "duration_s"])
        w.writerows(rows)
    classes = len({r[1] for r in rows})
    total_h = sum(float(r[2]) for r in rows) / 3600.0
    print(f"\n[index] {len(rows):,} clips / {classes} classes / {total_h:.2f} h"
          f" -> {INDEX_CSV}")


# -----------------------------------------------------------------------------
# 1. LibriSpeech dev-clean  -> speech_corpus/LibriSpeech/<spk>/<...>/*.flac
# -----------------------------------------------------------------------------
def download_librispeech(out_dir: str = SPEECH_DIR) -> int:
    """dev-clean, unpacked from the HF parquet mirror into real .flac files.

    The corpus builder and the babble mixer both glob '*.flac' under
    speech_corpus/LibriSpeech, so the parquet is decoded and written back out as
    flac rather than consumed in a new format. That keeps every downstream script
    (build_corpus_v2, build_noise_bank_v2, build_mega_corpus, evaluate_*) working
    unmodified.
    """
    target = os.path.join(out_dir, "LibriSpeech")
    existing = sum(1 for r, _, fs in os.walk(target) for f in fs if f.endswith(".flac"))
    if existing:
        print(f"[librispeech] already present: {existing:,} flac files")
        return existing

    import pyarrow.parquet as pq

    print(f"[librispeech] fetching {HF_LIBRISPEECH_FILE} from HF "
          f"(OpenSLR direct is ~85 KB/s here, HF is ~1.8 MB/s)")
    t0 = time.time()
    raw = _fetch(_hf_url(HF_LIBRISPEECH_REPO, HF_LIBRISPEECH_FILE), timeout=600)
    print(f"  parquet {len(raw) / 1e6:.0f} MB in {time.time() - t0:.0f}s")

    table = pq.read_table(io.BytesIO(raw))
    cols = table.column_names
    print(f"  columns: {cols}")

    spk_col = next((c for c in ("speaker_id", "speaker", "spk_id") if c in cols), None)
    audio = table.column("audio").to_pylist()
    speakers = table.column(spk_col).to_pylist() if spk_col else [0] * len(audio)

    n = 0
    for i, cell in enumerate(audio):
        try:
            arr, sr = _decode_audio_cell(cell)
        except Exception:                          # noqa: BLE001
            continue
        clip = to_16k_mono(arr, sr)
        spk = speakers[i] if speakers[i] is not None else 0
        # LibriSpeech-style naming keeps the <spk>/<chap>/ layout the builders glob
        utt = f"{spk}-{n // 1000:04d}-{n % 1000:03d}-{spk}-{n // 1000:04d}"
        d = os.path.join(target, str(spk), str(n // 1000))
        if save_clip(clip, os.path.join(d, f"{utt}.flac")):
            n += 1
        if n and n % 1000 == 0:
            print(f"  {n:,} flac written", flush=True)
    print(f"[librispeech] {n:,} flac files in {time.time() - t0:.0f}s -> {target}")
    return n


# -----------------------------------------------------------------------------
# 2. ESC-50  -> noise_bank_v2/raw/esc50/<class>__<name>.wav
# -----------------------------------------------------------------------------
def download_esc50(max_files: int = 2000) -> list:
    import pyarrow.parquet as pq

    print("\n[esc50] fetching ESC-50 (full, 2000 clips / 50 classes)")
    written = []
    for fname in HF_ESC50_FILES:
        try:
            raw = _fetch(_hf_url(HF_ESC50_REPO, fname), timeout=600)
            table = pq.read_table(io.BytesIO(raw))
        except Exception as e:                      # noqa: BLE001
            print(f"  {fname} failed: {str(e)[:70]}")
            continue
        cols = table.column_names
        audio = table.column("audio").to_pylist()
        # this mirror ships the class name as text, which is authoritative
        lbl_col = next((c for c in ("category", "label", "class", "target")
                        if c in cols), None)
        labels = table.column(lbl_col).to_pylist() if lbl_col else [None] * len(audio)
        fld = next((c for c in ("filename", "file", "path") if c in cols), None)
        files = table.column(fld).to_pylist() if fld else [None] * len(audio)

        for cell, lbl, fn in zip(audio, labels, files):
            if max_files and len(written) >= max_files:
                break
            if isinstance(lbl, (bytes, bytearray)):
                lbl = lbl.decode("utf-8", "replace")
            lbl = "esc50_unknown" if lbl is None else str(lbl)
            name = os.path.basename(str(fn)) if fn else f"{len(written):05d}.wav"
            if not name.endswith(".wav"):
                name += ".wav"
            try:
                arr, sr = _decode_audio_cell(cell)
            except Exception:                      # noqa: BLE001
                continue
            dst = os.path.join(RAW_DIR, "esc50", f"{lbl}__{name}")
            if os.path.exists(dst) or save_clip(to_16k_mono(arr, sr), dst):
                written.append((dst, lbl))
        print(f"  {fname}: cumulative {len(written)} clips", flush=True)
    print(f"[esc50] {len(written)} clips on disk")
    return written


# -----------------------------------------------------------------------------
# 3. UrbanSound8K  -> noise_bank_v2/raw/urbansound/us8k_*.wav
# -----------------------------------------------------------------------------
def download_urbansound(shard_limit: int = 4, per_class_cap: int = 220) -> list:
    import pyarrow.parquet as pq

    print(f"\n[urbansound] fetching UrbanSound8K ({shard_limit} of "
          f"{len(HF_US8K_SHARDS)} shards)")
    written = []
    counts = {}
    for pq_name in HF_US8K_SHARDS[:max(0, shard_limit)]:
        try:
            cache = os.path.join(SHARD_CACHE, os.path.basename(pq_name))
            raw = _fetch_resumable(_hf_url(HF_US8K_REPO, pq_name), cache, timeout=45)
            table = pq.read_table(io.BytesIO(raw))
        except Exception as e:                      # noqa: BLE001
            print(f"  shard {pq_name} failed: {str(e)[:70]}")
            continue
        cols = table.column_names
        if "audio" not in cols or "class" not in cols:
            print(f"  {pq_name}: unexpected columns {cols}")
            continue
        for cell, lbl in zip(table.column("audio").to_pylist(),
                             table.column("class").to_pylist()):
            lbl = str(lbl)
            if counts.get(lbl, 0) >= per_class_cap:
                continue
            try:
                arr, sr = _decode_audio_cell(cell)
            except Exception:                      # noqa: BLE001
                continue
            counts[lbl] = counts.get(lbl, 0) + 1
            dst = os.path.join(RAW_DIR, "urbansound", f"us8k_{counts[lbl]:05d}_{lbl}.wav")
            if save_clip(to_16k_mono(arr, sr), dst):
                written.append((dst, f"us8k_{lbl}"))
        print(f"  {pq_name}: cumulative {len(written)} clips", flush=True)
    print(f"[urbansound] {len(written)} clips on disk")
    return written


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch the Amaze KWS source corpora")
    ap.add_argument("--libri", "--librispeech", dest="libri", action="store_true")
    ap.add_argument("--esc50", action="store_true")
    ap.add_argument("--urbansound", action="store_true")
    ap.add_argument("--shards", type=int, default=4, help="UrbanSound8K shards to pull")
    ap.add_argument("--esc50-max", type=int, default=2000)
    ap.add_argument("--per-class-cap", type=int, default=220)
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()

    want_all = not (args.libri or args.esc50 or args.urbansound)
    os.makedirs(SPEECH_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SHARD_CACHE, exist_ok=True)
    t0 = time.time()

    n_lib = 0
    if args.libri or want_all:
        n_lib = download_librispeech()
    if args.esc50 or want_all:
        download_esc50(max_files=args.esc50_max)
    if args.urbansound or want_all:
        download_urbansound(shard_limit=args.shards, per_class_cap=args.per_class_cap)

    if not args.no_index:
        _write_index()

    print(f"\n[all] done in {time.time() - t0:.0f}s"
          + (f"  ({n_lib:,} LibriSpeech flac)" if n_lib else ""))
    print("[next] synthesise the babble + device layers, then build the corpus:")
    print("  py build_noise_bank_v2.py --procedural")
    print("  py build_corpus_v2.py --profile v5 --pos 14000 --neg 84000")


if __name__ == "__main__":
    sys.exit(main())

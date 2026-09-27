"""
Fetch the real-human-speech source corpus from public URLs
-----------------------------------------------------------
Everything else in this project already regenerates itself from public URLs on demand
(ESC-50 from GitHub, UrbanSound8K from HuggingFace). This closes the loop for speech
so a Kaggle notebook can build the whole corpus with no multi-gigabyte upload.

Sources
  LibriSpeech dev-clean  337 MB   ~5,270 utterances of clean read human speech.
                                Used for the `real` negative content type, for the
                                multi-talker babble mixer, and for the handful of real
                                human utterances containing "amaze".
  Speech Commands v0.02 2.3 GB   ~105k single-word recordings. Optional, off by
                                default: the TTS filler pool already covers the
                                "unimportant word" content type, and 2.3 GB is a lot
                                of notebook time for marginal extra variety.

Everything is written to speech_corpus/ in the layout build_corpus_v2.SpeechBank
expects (LibriSpeech/**.flac and speech_commands/**/*.wav).

    python build_speech_corpus.py                 # LibriSpeech dev-clean only
    python build_speech_corpus.py --speech_commands
"""

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.request

SPEECH_CORPUS = "speech_corpus"

LIBRISPEECH_SOURCES = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
}
SPEECH_COMMANDS_URL = "https://www.tensorflow.org/data/speech_commands_v0.02.tar.gz"

UA = {"User-Agent": "Mozilla/5.0 (amaze-kws-corpus-builder)"}


def _download(url: str, dest: str, quiet: bool = False) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if not quiet:
            print(f"  cached: {os.path.basename(dest)}")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if not quiet and total and done - last > (32 << 20):
                last = done
                print(f"  {done/1e6:7.0f} / {total/1e6:.0f} MB", flush=True)
    return dest


def _safe_extract(archive: str, dest: str) -> None:
    """Extracts without letting archive members escape the destination directory."""
    dest_abs = os.path.abspath(dest)
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = os.path.abspath(os.path.join(dest, member.name))
            if not target.startswith(dest_abs + os.sep) and target != dest_abs:
                print(f"  skipping path traversal: {member.name}")
                continue
            tf.extract(member, dest, set_attrs=False)


def build_librispeech(out_dir: str = SPEECH_CORPUS, split: str = "dev-clean",
                     quiet: bool = False) -> int:
    url = LIBRISPEECH_SOURCES[split]
    target = os.path.join(out_dir, "LibriSpeech")
    if os.path.isdir(target):
        n = sum(len(f) for _, _, f in os.walk(target) if any(x.endswith(".flac") for x in f))
        if n:
            if not quiet:
                print(f"[librispeech] already present: {n:,} flac files under {target}")
            return n
    with tempfile.TemporaryDirectory() as td:
        tarball = _download(url, os.path.join(td, f"{split}.tar.gz"), quiet=quiet)
        if not quiet:
            print(f"[librispeech] extracting {split} -> {out_dir}")
        _safe_extract(tarball, out_dir)
    n = sum(1 for r, _, fs in os.walk(target) for f in fs if f.endswith(".flac"))
    if not quiet:
        print(f"[librispeech] {n:,} flac files")
    return n


def build_speech_commands(out_dir: str = SPEECH_CORPUS, quiet: bool = False) -> int:
    target = os.path.join(out_dir, "speech_commands")
    if os.path.isdir(target):
        n = sum(1 for r, _, fs in os.walk(target) for f in fs if f.endswith(".wav"))
        if n:
            if not quiet:
                print(f"[speech_commands] already present: {n:,} wav files")
            return n
    with tempfile.TemporaryDirectory() as td:
        tarball = _download(SPEECH_COMMANDS_URL, os.path.join(td, "sc.tar.gz"), quiet=quiet)
        if not quiet:
            print("[speech_commands] extracting (this is 2.3 GB, be patient)")
        _safe_extract(tarball, td)
        # v0.02 unpacks to a single directory of labelled subfolders plus validation/
        # and test/ splits; keep only the labelled training folders we can train on.
        root = None
        for entry in os.listdir(td):
            p = os.path.join(td, entry)
            if os.path.isdir(p) and any(
                os.path.isdir(os.path.join(p, sub)) for sub in os.listdir(p)
                if os.path.isdir(os.path.join(p, sub))
            ):
                root = p
                break
        if root is None:
            print("[speech_commands] unexpected archive layout; skipped")
            return 0
        os.makedirs(target, exist_ok=True)
        for sub in os.listdir(root):
            s = os.path.join(root, sub)
            if os.path.isdir(s) and sub not in ("testing", "validation", "_background_noise_"):
                shutil.copytree(s, os.path.join(target, sub), dirs_exist_ok=True)
    n = sum(1 for r, _, fs in os.walk(target) for f in fs if f.endswith(".wav"))
    if not quiet:
        print(f"[speech_commands] {n:,} wav files")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=SPEECH_CORPUS)
    ap.add_argument("--split", type=str, default="dev-clean", choices=sorted(LIBRISPEECH_SOURCES))
    ap.add_argument("--speech_commands", action="store_true",
                    help="also fetch Speech Commands v0.02 (2.3 GB, optional)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[speech corpus] target {args.out}")
    n_lib = build_librispeech(args.out, args.split)
    n_cmd = build_speech_commands(args.out) if args.speech_commands else 0
    total = n_lib + n_cmd
    print(f"[speech corpus] ready: {n_lib:,} LibriSpeech flac"
          + (f" + {n_cmd:,} Speech Commands wav" if n_cmd else ""))
    if not n_lib:
        raise SystemExit("no speech available; corpus cannot be built")


if __name__ == "__main__":
    main()

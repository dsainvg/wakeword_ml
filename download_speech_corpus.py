"""
Downloads and prepares diverse real human speech corpora from LibriSpeech & OpenSLR
to form a massive, unrepeated negative human speech pool (thousands of real speakers).
"""

import os
import sys
import tarfile
import urllib.request
import time

CORPUS_DIR = "speech_corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

DATASETS = [
    ("dev-clean.tar.gz", "http://www.openslr.org/resources/12/dev-clean.tar.gz", "LibriSpeech dev-clean (5.4 hrs, 40 speakers)"),
    ("dev-other.tar.gz", "http://www.openslr.org/resources/12/dev-other.tar.gz", "LibriSpeech dev-other (5.3 hrs, 40 speakers, noisy/accented)"),
    ("test-clean.tar.gz", "http://www.openslr.org/resources/12/test-clean.tar.gz", "LibriSpeech test-clean (5.4 hrs, 40 speakers)"),
]

def download_file(url: str, dest_path: str, desc: str):
    if os.path.exists(dest_path):
        print(f"[EXISTS] {dest_path} already downloaded.", flush=True)
        return

    print(f"Downloading {desc}...", flush=True)
    print(f"  URL : {url}", flush=True)
    print(f"  Dest: {dest_path}", flush=True)

    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total_len = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        last_print = 0

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_print > 5.0 or downloaded == total_len:
                pct = (downloaded / total_len * 100) if total_len else 0
                mb = downloaded / (1024 * 1024)
                rate = mb / (time.time() - t0 + 1e-6)
                print(f"  -> {mb:.1f} MB / {total_len/(1024*1024):.1f} MB ({pct:.1f}%) @ {rate:.2f} MB/s", flush=True)
                last_print = time.time()

    print(f"  Download finished in {time.time() - t0:.1f}s!\n", flush=True)

def extract_tar(tar_path: str, extract_to: str):
    print(f"Extracting {tar_path} into {extract_to}...", flush=True)
    t0 = time.time()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)
    print(f"  Extracted in {time.time() - t0:.1f}s!\n", flush=True)

def main():
    print("=" * 65)
    print(" Downloading Large-Scale Real Human Speech Corpora (LibriSpeech)")
    print("=" * 65)

    for fname, url, desc in DATASETS:
        tar_path = os.path.join(CORPUS_DIR, fname)
        try:
            download_file(url, tar_path, desc)
            extract_tar(tar_path, CORPUS_DIR)
            # Remove tar.gz to save disk space after extraction
            if os.path.exists(tar_path):
                os.remove(tar_path)
                print(f"  Cleaned up archive {fname} to conserve disk space.\n", flush=True)
        except Exception as e:
            print(f"[Error] Failed processing {fname}: {e}", flush=True)

    # Count extracted files
    flac_files = []
    for root, _, files in os.walk(CORPUS_DIR):
        for f in files:
            if f.endswith(".flac"):
                flac_files.append(os.path.join(root, f))

    print("=" * 65)
    print(f" Corpus Ready! Total Unique Real Human Speech Recordings: {len(flac_files):,}")
    print("=" * 65)

if __name__ == "__main__":
    main()

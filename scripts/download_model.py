"""
download_model.py
-----------------
Robust chunked downloader for transformer weights and tokenizer files.
"""

import os
import sys
import time
import urllib.request

MODEL_FILES = [
    ("config.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"),
    ("tokenizer.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"),
    ("tokenizer_config.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"),
    ("vocab.txt", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"),
    ("special_tokens_map.json", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json"),
    ("model.safetensors", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"),
]


def download_file(url: str, dest_path: str):
    print(f"Downloading {os.path.basename(dest_path)}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 512  # 512 KB
        start_time = time.time()

        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                speed = downloaded / (time.time() - start_time + 1e-5) / 1024
                print(f"\r  {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({pct:.1f}%) - {speed:.1f} KB/s", end="", flush=True)
        print()


def main():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "transformer")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: {target_dir}")

    for fname, url in MODEL_FILES:
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            print(f"Skipping {fname} (already exists, size: {os.path.getsize(fpath)} bytes)")
            continue
        download_file(url, fpath)

    print("All model files downloaded successfully!")


if __name__ == "__main__":
    main()

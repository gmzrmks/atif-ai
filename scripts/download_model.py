"""SciBERT dosyalarını doğrudan URL'den indirir (SSL verify=False).
Hedef: ./models/scibert/ → notebook bunu base_model olarak kullanır.

Bir kez çalıştır.
"""
import os, ssl, sys
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests

BASE = "https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main"
FILES = [
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",          # opsiyonel (fast tokenizer)
    "special_tokens_map.json", # opsiyonel
    "pytorch_model.bin",
]

OUT = Path(__file__).parent.parent / "models" / "scibert"
OUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) huggingface_hub/manual",
})

for fname in FILES:
    target = OUT / fname
    if target.exists() and target.stat().st_size > 0:
        print(f"  [skip] {fname} ({target.stat().st_size/1e6:.1f} MB)")
        continue
    url = f"{BASE}/{fname}"
    print(f"  [get ] {fname}", flush=True)
    try:
        r = session.get(url, stream=True, timeout=60, allow_redirects=True)
        if r.status_code == 404:
            print(f"  [404 ] {fname} - skipped")
            continue
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        sys.stdout.write(f"\r        {pct:5.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)")
                        sys.stdout.flush()
        sys.stdout.write("\r" + " " * 60 + "\r")
        print(f"  [ok  ] {fname} ({downloaded/1e6:.1f} MB)")
    except Exception as e:
        print(f"  [fail] {fname}: {type(e).__name__}: {e}")
        if fname in ("config.json", "vocab.txt", "tokenizer_config.json", "pytorch_model.bin"):
            sys.exit(1)

print(f"\nBitti. Yerel SciBERT yolu: {OUT.resolve()}")
print("Notebook'ta CONFIG['base_model'] zaten string accept ediyor — yerel path geç.")

"""Pre-download HF datasets so notebook'un ilk run'ı hızlı olsun.

Çalıştır:
    .venv\\Scripts\\python.exe scripts\\download_data.py
"""
from datasets import load_dataset
import os, time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "hf")
os.makedirs(CACHE_DIR, exist_ok=True)

SOURCES = [
    ("scinli",         "tasksource/scinli",     None),
    ("multi_nli",      "nyu-mll/multi_nli",     None),
    ("fever_nli",      "pietrolesci/nli_fever", None),
    ("scifact_claims", "allenai/scifact",       "claims"),
    ("scifact_corpus", "allenai/scifact",       "corpus"),
]

for name, ds_id, cfg in SOURCES:
    t0 = time.time()
    print(f"[{name}] indiriliyor: {ds_id} (cfg={cfg})")
    try:
        ds = load_dataset(ds_id, cfg, cache_dir=CACHE_DIR, trust_remote_code=True) if cfg \
             else load_dataset(ds_id, cache_dir=CACHE_DIR, trust_remote_code=True)
        sizes = {s: len(ds[s]) for s in ds}
        print(f"  OK ({time.time()-t0:.0f}s): {sizes}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

print("Bitti.")

"""Bilimsel Çelişki Tespit — Flask web servisi.

İki PDF veya TXT makalesi yüklenir, seçilen model her bir makaledeki cümleleri
karşılıklı eşler, çelişki/destek skorunu döndürür.

Modeller:
  - general    -> models/contradiction_model/      (karışık NLI + SciFact adapt)
  - scientific -> models/contradiction_model_sci/  (sadece SciNLI + SciFact adapt)

Çalıştır:
    .venv\\Scripts\\python.exe app\\app.py
Sonra tarayıcıdan: http://localhost:5000
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "scibert"
MAX_LEN = 256
ID2LABEL = {0: "SUPPORT", 1: "NEUTRAL", 2: "CONTRADICT"}

# Mevcut modeller: key -> (display name, klasör)
MODELS_AVAILABLE: Dict[str, Dict[str, str]] = {
    "general": {
        "label": "Genel NLI",
        "dir": str(ROOT / "models" / "contradiction_model"),
    },
    "scientific": {
        "label": "Bilimsel",
        "dir": str(ROOT / "models" / "contradiction_model_sci"),
    },
}


class ContradictionDetector(nn.Module):
    def __init__(self, base, n_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(base))
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, n_labels))

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kw):
        kw_in = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw_in["token_type_ids"] = token_type_ids
        out = self.encoder(**kw_in)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def _load_one(model_dir: Path):
    weights = model_dir / "pytorch_model.bin"
    if not weights.exists():
        raise RuntimeError(
            f"Eğitilmiş model bulunamadı: {weights}\n"
            "Önce ilgili notebook'u çalıştırıp eğitimi tamamlayın."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContradictionDetector(BASE_MODEL)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    # Tokenizer adapte model klasöründe kayıtlı (tokenizer.save_pretrained)
    tok_dir = model_dir if (model_dir / "tokenizer.json").exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    return model, tokenizer, device


# Lazy cache: aynı modeli iki kez yüklemeyelim
_cache: Dict[str, tuple] = {}

def get_model(key: str):
    if key not in MODELS_AVAILABLE:
        raise ValueError(f"Bilinmeyen model: {key}. Seçenekler: {list(MODELS_AVAILABLE)}")
    if key not in _cache:
        info = MODELS_AVAILABLE[key]
        _cache[key] = _load_one(Path(info["dir"]))
    return _cache[key]


# -------- Metin çıkarma & cümle bölme --------

def extract_text(file_storage) -> str:
    name = (file_storage.filename or "").lower()
    data = file_storage.read()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZĞÜŞİÖÇ])")

def split_sentences(text: str, min_len: int = 30, max_n: int = 60) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in _SENT_RE.split(text)]
    sents = [s for s in sents if min_len <= len(s) <= 600]
    return sents[:max_n]


# -------- Inference --------

@torch.no_grad()
def score_pairs(pairs: List[Tuple[str, str]], model_key: str, batch_size: int = 32):
    model, tok, device = get_model(model_key)
    out = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        a = [p[0] for p in batch]
        b = [p[1] for p in batch]
        enc = tok(a, b, max_length=MAX_LEN, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        out.extend(probs.tolist())
    return out


def analyze(article_a: str, article_b: str, model_key: str, top_k: int = 5) -> dict:
    sents_a = split_sentences(article_a)
    sents_b = split_sentences(article_b)
    if not sents_a or not sents_b:
        return {"error": "Makalelerden anlamlı cümle çıkarılamadı."}

    pairs = [(a, b) for a in sents_a for b in sents_b]
    probs = score_pairs(pairs, model_key)

    contra_scores = [(p[2] + 0.5 * p[1], p, a, b)
                     for p, (a, b) in zip(probs, pairs)]
    contra_scores.sort(key=lambda x: -x[0])

    avg = [sum(x) / len(x) for x in zip(*probs)]
    overall_pred = int(max(range(3), key=lambda i: avg[i]))
    overall_score = avg[2] + 0.5 * avg[1]

    top = []
    for score, p, a, b in contra_scores[:top_k]:
        top.append({
            "score": round(score, 4),
            "label": ID2LABEL[int(max(range(3), key=lambda i: p[i]))],
            "p_support": round(p[0], 3),
            "p_neutral": round(p[1], 3),
            "p_contradict": round(p[2], 3),
            "sentence_a": a,
            "sentence_b": b,
        })

    return {
        "model": model_key,
        "model_label": MODELS_AVAILABLE[model_key]["label"],
        "n_sentences_a": len(sents_a),
        "n_sentences_b": len(sents_b),
        "n_pairs": len(pairs),
        "overall_label": ID2LABEL[overall_pred],
        "overall_score": round(overall_score, 4),
        "avg_p_support": round(avg[0], 4),
        "avg_p_neutral": round(avg[1], 4),
        "avg_p_contradict": round(avg[2], 4),
        "top_contradictions": top,
    }


# -------- Flask --------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def _model_status():
    """Hangi modeller diskte mevcut, hangileri eksik?"""
    out = []
    for key, info in MODELS_AVAILABLE.items():
        weights = Path(info["dir"]) / "pytorch_model.bin"
        out.append({
            "key": key,
            "label": info["label"],
            "available": weights.exists(),
            "size_mb": round(weights.stat().st_size / 1e6, 1) if weights.exists() else 0,
        })
    return out


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", models=_model_status())


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(_model_status())


@app.route("/api/compare", methods=["POST"])
def compare():
    if "file_a" not in request.files or "file_b" not in request.files:
        return jsonify({"error": "İki dosya da yükleyin (file_a + file_b)."}), 400
    fa = request.files["file_a"]
    fb = request.files["file_b"]
    if not fa.filename or not fb.filename:
        return jsonify({"error": "Boş dosya."}), 400

    model_key = request.form.get("model", "general")
    if model_key not in MODELS_AVAILABLE:
        return jsonify({"error": f"Bilinmeyen model: {model_key}"}), 400

    try:
        text_a = extract_text(fa)
        text_b = extract_text(fb)
        result = analyze(text_a, text_b, model_key)
        result["file_a"] = fa.filename
        result["file_b"] = fb.filename
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    print("Mevcut modeller:")
    for m in _model_status():
        mark = "OK" if m["available"] else "EKSIK"
        size = f"{m['size_mb']:.1f} MB" if m["available"] else "-"
        print(f"  [{mark}] {m['key']:<12} {m['label']} ({size})")
    print("\nServis: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

"""SciFact corpus'undan ground-truth etiketli test çiftleri üret.

Her çift için iki PDF: abstract A (cited doc) + claim'in genişletilmiş hâli (mini paper B).
Dosya isimleri beklenen ilişkiyi gösterir. Web arayüzünde test için kullanılabilir.

Çıktı: ./samples/  (gitignored)
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from pathlib import Path
from datasets import load_dataset
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / "cache" / "hf"

print("SciFact yükleniyor...")
claims = load_dataset("allenai/scifact", "claims",
                      cache_dir=str(CACHE), trust_remote_code=True)["validation"]
corpus = load_dataset("allenai/scifact", "corpus",
                      cache_dir=str(CACHE), trust_remote_code=True)["train"]
doc_lookup = {int(r["doc_id"]): (r.get("title", ""), " ".join(r["abstract"]))
              for r in corpus}

# Etiket başına ilk N adet kullanışlı (uzun abstract'lı) örnek seç
support, contradict = [], []
for ex in claims:
    label = (ex.get("evidence_label") or "").strip()
    doc_id = (ex.get("evidence_doc_id") or "").strip()
    if not doc_id or label not in ("SUPPORT", "CONTRADICT"):
        continue
    title, abstract = doc_lookup.get(int(doc_id), ("", ""))
    if len(abstract) < 600:  # çok kısa olanları geç
        continue
    item = {
        "claim_id": ex["id"],
        "claim": ex["claim"],
        "doc_id": doc_id,
        "title": title or f"Document {doc_id}",
        "abstract": abstract,
    }
    if label == "SUPPORT":
        support.append(item)
    else:
        contradict.append(item)

print(f"Aday SUPPORT: {len(support)}, CONTRADICT: {len(contradict)}")

# Her etiketten 3'er örnek
selected = []
for i, item in enumerate(support[:3]):
    selected.append(("SUPPORT", i + 1, item))
for i, item in enumerate(contradict[:3]):
    selected.append(("CONTRADICT", i + 1, item))

def make_pdf(path: Path, title: str, body: str):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_st = ParagraphStyle("t", parent=styles["Heading2"],
                              fontSize=14, spaceAfter=14)
    body_st = ParagraphStyle("b", parent=styles["BodyText"],
                             fontSize=11, leading=15, spaceAfter=8)
    flow = [Paragraph(title, title_st), Spacer(1, 6)]
    for para in body.split("\n\n"):
        flow.append(Paragraph(para.strip(), body_st))
    doc.build(flow)

# Bir pseudo-paper B üreteyim: claim'i + abstract bilgisini tutarlı/çelişen şekilde sun
def build_paper_b(claim: str, label: str) -> str:
    if label == "SUPPORT":
        intro = f"This study supports the following hypothesis: {claim}"
        body = (f"Our investigation provides strong evidence consistent with the claim that {claim.lower()}. "
                f"Multiple experimental conditions confirmed this observation. "
                f"The data shows that {claim.lower()} The findings align with prior literature in this area. "
                f"We conclude that {claim.lower()}")
    else:  # CONTRADICT
        intro = f"This study challenges the claim that {claim}"
        body = (f"Contrary to the assertion that {claim.lower()}, our results indicate the opposite trend. "
                f"We did not observe the effect implied by the claim. "
                f"In fact, our analyses suggest that {claim.lower()} is not supported by the data. "
                f"The findings argue against the hypothesis that {claim.lower()}")
    return intro + "\n\n" + body

print(f"\nSeçilen {len(selected)} çift, PDF üretiliyor...\n")
for label, idx, item in selected:
    base = f"{idx:02d}_{label.lower()}_claim{item['claim_id']}"
    pa = OUT / f"{base}_A_paper.pdf"
    pb = OUT / f"{base}_B_paper.pdf"

    # Paper A: gerçek abstract (SciFact corpus)
    make_pdf(pa, f"Paper A: {item['title'][:80]}", item["abstract"])
    # Paper B: claim'in label'a göre genişletilmiş kısa metni
    body_b = build_paper_b(item["claim"], label)
    make_pdf(pb, f"Paper B: Study on '{item['claim'][:60]}...'", body_b)

    print(f"  [{label:10}] {pa.name}")
    print(f"             {pb.name}")
    print(f"             claim: {item['claim'][:90]}")
    print()

print(f"Toplam {len(selected) * 2} PDF kaydedildi: {OUT}")
print("\nFlask UI'de A ve B dosyalarını birlikte yükleyip etiketin doğruluğunu test edebilirsin.")
print("Dosya adı SUPPORT diyorsa model SUPPORT/CONTRADICT bekleniyor; CONTRADICT diyorsa CONTRADICT.")

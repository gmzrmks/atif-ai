# Bilimsel Çelişki Tespit Sistemi

İki bilimsel metin (cümle, abstract veya tam makale) arasındaki ilişkiyi
**SUPPORT / NEUTRAL / CONTRADICT** olarak sınıflandıran SciBERT tabanlı bir model.
Ek olarak `[0, 1]` aralığında skalar bir çelişki skoru üretir:
`score = P(contradict) + 0.5·P(neutral)`.

İki kullanım modu:
- **Notebook** (`contradiction_detection.ipynb`) — veri hazırlama, eğitim, domain adaptation ve değerlendirme.
- **Flask web servisi** (`app/app.py`) — eğitilmiş modeli yükler, iki PDF/TXT makalesi alır, cümle-bazlı analiz sonucunu tarayıcıda gösterir.

---

## 1. Yaklaşım

Doğal Dil Çıkarımı (NLI) klasik bir görevdir: bir "öncül" (premise) ile bir
"hipotez" (hypothesis) arasındaki mantıksal ilişkiyi tahmin etmek. Üç sınıf:

| Sınıf | Anlamı |
|---|---|
| SUPPORT (entailment) | Öncül, hipotezi mantıksal olarak doğrular |
| NEUTRAL | Yetersiz bilgi / ilişki belirsiz |
| CONTRADICT (contradiction) | Öncül ve hipotez çelişir |

Bu projede üç farklı NLI veri seti birleştirilip ortak 3-sınıf etiketine indirgendi,
SciBERT (bilimsel literatürde önceden eğitilmiş BERT varyantı) üstüne 3-sınıflı
bir sınıflandırma başlığı ile fine-tune edildi. Ardından SciFact'in kendi train
seti üzerinde domain adaptation uygulanarak bilimsel makale alanına özelleştirildi.

## 2. Veri Setleri

Tüm sayılar **(premise, hypothesis) çift** sayısıdır — makale veya cümle değil.
Her çift iki cümleden oluşur (öncül + hipotez) ve bir etiket taşır.

| Kaynak | HF id | Çift | Domain | Rol |
|---|---|---|---|---|
| **SciNLI** | `tasksource/scinli` | 101 412 | Bilimsel makale (ACL Anthology'den çıkarılmış cümle çiftleri) | Train |
| **FEVER-NLI** | `pietrolesci/nli_fever` | 208 346 | Wikipedia factual claims | Train |
| **MultiNLI** | `nyu-mll/multi_nli` | 250 000 (stratified) | Karışık metin türü | Train |
| **SciFact (train)** | `allenai/scifact` | 1 463 | Bilimsel claim–abstract | Domain adaptation |
| **SciFact (validation)** | `allenai/scifact` | 519 | Bilimsel claim–abstract | **Test** |

Birleşik eğitim korpusu: **~559 563 çift**. SNLI (Flickr fotoğraf altyazıları) bilimsel
domain'le alakası olmadığı için dahil edilmedi.

### Etiket normalizasyonu (kaynak → ortak 3-sınıf)

| Kaynak | Birleşik |
|---|---|
| entailment / SUPPORTS | 0 = SUPPORT |
| neutral / reasoning (SciNLI) / NOT ENOUGH INFO | 1 = NEUTRAL |
| contradiction / contrasting (SciNLI) / REFUTES | 2 = CONTRADICT |

## 3. Model

- **Base:** `allenai/scibert_scivocab_uncased` (110M parametre, bilimsel vocab=31090)
- **Head:** `Dropout(0.1) → Linear(768, 3)`
- **Loss:** CrossEntropy
- **Input format:** `[CLS] premise [SEP] hypothesis [SEP]`, max_length=256
- **Optimizer:** AdamW, lr=2e-5 (ana eğitim) → lr=1e-5 (adaptation)

## 4. Eğitim Pipeline'ı

```
1. Veri yükleme       SciNLI + FEVER-NLI + MultiNLI → ortak 3-sınıf
2. Tokenization       SciBERT tokenizer, max_len=256, padding=max
3. Fine-tune          2 epoch, batch=32, fp16, RTX 5080 (~2 saat)
4. Test (genel)       SciFact validation üzerinde değerlendirme
5. Yedek              Adaptation öncesi modeli ayrı klasöre kopyala
6. Domain adaptation  SciFact train (1261) ile lr=1e-5, 3 epoch (~30 sn)
7. Test (adapte)      SciFact üzerinde tekrar değerlendirme
8. Kayıt              Final modeli models/contradiction_model/ altına yaz
```

## 5. Sonuçlar (SciFact test, 519 çift)

| Metrik | Adaptation öncesi | Adaptation sonrası | Δ |
|---|---|---|---|
| Accuracy | 0.559 | **0.684** | +12.5 puan |
| F1 macro | 0.550 | **0.663** | +11.3 puan |
| F1 (CONTRADICT) | 0.494 | **0.530** | +3.6 puan |
| R² | 0.167 | −0.010 | −0.18 |

Domain adaptation, ana NLI eğitiminin üstüne **+12 puan accuracy / +11 puan F1**
kazandırdı. Sonuç, SciFact leaderboard'unda SciBERT-base ile elde edilen tipik banda
(0.65–0.75) yerleşiyor.

> **R² hakkında:** Skalar skor hedefi yalnızca 3 ayrık değer alır (0 / 0.5 / 1);
> R² bu yapıda çok sert ve yanıltıcı bir metriktir (sıfıra yakın çıkar). Asıl
> değerlendirme sınıflandırma metrikleridir (accuracy / F1 macro). Sınıf bazında:
> SUPPORT F1=0.73, NEUTRAL F1=0.73, CONTRADICT F1=0.53.

## 6. Kurulum ve Çalıştırma

### Ön koşullar
- Python 3.11
- CUDA 12.8+ destekli NVIDIA GPU (RTX 5080 önerilen)
- Windows 11 / Linux

### Klonlama
```bash
git clone https://github.com/gmzrmks/atif-ai.git
```

Eğitilmiş model ağırlıkları (`*.bin`) Git LFS üzerinden saklanıyor, klonlama sırasında otomatik indirilir.

Alternatif olarak notebook çalıştırılırsa (~2 saat) ağırlıklar yeniden oluşur.

### Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate                           # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Veri ve base model
```bash
python scripts/download_data.py                  # NLI veri setleri + SciFact
python scripts/download_model.py                 # SciBERT base ağırlıkları
```

### Eğitim
```bash
jupyter notebook contradiction_detection.ipynb
# Kernel → Restart → Run All  (~2 saat)
```

Tamamlanınca `models/contradiction_model/pytorch_model.bin` (440 MB) oluşur.

### Web servisi
```bash
python app/app.py
# Tarayıcıdan: http://localhost:5000
```

İki dosya (PDF veya TXT, max 25 MB) yükle, "Karşılaştır":
- Genel verdict + skor
- SUPPORT / NEUTRAL / CONTRADICT için ortalama olasılıklar
- En yüksek çelişki sinyalli ilk 5 cümle çifti

### Örnek test dosyaları
```bash
python scripts/make_samples.py                   # samples/ altına 12 PDF üretir
```
Ground-truth etiketli SciFact pair'lerinden hazırlanmış test PDF'leri.

## 7. Dosya Yapısı

```
atif_ai/
├── contradiction_detection.ipynb     # Ana notebook (11 hücre)
├── app/
│   ├── app.py                         # Flask servisi
│   └── templates/index.html           # Web arayüzü
├── scripts/
│   ├── download_data.py               # Veri seti indirici
│   ├── download_model.py              # SciBERT pre-download (SSL bypass'lı)
│   └── make_samples.py                # Test PDF'leri üretir
├── samples/                           # Hazır test çiftleri (PDF)
├── cache/                             # (gitignore) Veri + tokenized cache
├── models/                            # (gitignore) Eğitilmiş modeller
├── requirements.txt
├── CLAUDE.md                          # Teknik dokümantasyon
└── README.md
```

## 8. Flask API

`POST /api/compare` — form-data: `file_a`, `file_b` (max 25 MB), `model` (opsiyonel)

```json
{
  "overall_label": "CONTRADICT",
  "overall_score": 0.71,
  "avg_p_support": 0.18,
  "avg_p_neutral": 0.20,
  "avg_p_contradict": 0.62,
  "n_pairs": 1800,
  "top_contradictions": [
    {
      "score": 0.94,
      "label": "CONTRADICT",
      "sentence_a": "...",
      "sentence_b": "...",
      "p_support": 0.04,
      "p_neutral": 0.02,
      "p_contradict": 0.94
    }
  ]
}
```

Sentence-level pairwise tarama: her makaleden ilk 60 anlamlı cümle alınır,
3600'e kadar çift modele verilir, ortalama olasılık ve top-K en yüksek çelişki
çifti döndürülür.

## Lisans

Akademik kullanım için. SciBERT: Apache 2.0 · SciNLI: BSD · SciFact: ODC-BY · MultiNLI: OANC · FEVER: CC-BY-SA.

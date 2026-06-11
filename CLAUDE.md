# Bilimsel Çelişki Tespit Sistemi

## GÖREV

İki cümle/abstract verildiğinde aralarındaki ilişkiyi **SUPPORT / NEUTRAL / CONTRADICT** olarak sınıflandıran SciBERT tabanlı bir NLP sistemi. Tek bir Jupyter notebook (`contradiction_detection.ipynb`) olarak organize. Veriler HuggingFace'ten otomatik gelir; ağ kısıtlıysa `scripts/` altındaki yardımcı script'lerle önceden indirilir, notebook offline modda çalışır.

## DONANIM

- NVIDIA RTX 5080 (16 GB VRAM, Blackwell)
- 32 GB RAM, ~5 GB boş disk
- Python 3.11, CUDA 12.8+, Windows 11

## YAKLAŞIM

Pseudo-labeling **yok**. Bilimsel domain'e öncelik veren üç etiketli NLI veri seti birleştirilip ortak 3-sınıf etiketine indirgenir, SciBERT bu birleşik korpus üzerinde supervised fine-tune edilir. Test SciFact üzerinde yapılır (makale claim/evidence çiftleri).

Süre/kalite dengesi:
- SciNLI (bilimsel) **tam** tutulur — domain match için kritik
- FEVER-NLI (Wikipedia factual) **tam** tutulur — çelişki sinyali güçlü
- MultiNLI'dan stratified ~250k alınır — karışım, dil çeşitliliği için
- SNLI (Flickr foto altyazıları) **dahil değil** — bilimsel iddia yapısıyla alakasız

## VERİ STRATEJİSİ

| Kaynak | HF id | Train (kullanılan) | Domain | Rol |
|---|---|---|---|---|
| **SciNLI** | `tasksource/scinli` | 101k (tam) | **Bilimsel makale (ACL Anthology)** | Train + val |
| FEVER-NLI | `pietrolesci/nli_fever` | 208k (tam) | Wikipedia factual claims | Train + val |
| MultiNLI | `nyu-mll/multi_nli` | 250k (stratified) | Karışık metin türü | Train + val |
| SciFact | `allenai/scifact` (`trust_remote_code=True`) | — | Bilimsel makale abstract'ları | **Test** |

**Final eğitim korpusu:** ~559k çift (108 MB parquet cache).

### Etiket normalizasyonu (4-sınıf → 3-sınıf)

| Kaynak etiketi | Birleşik |
|---|---|
| entailment / SUPPORTS / 0 | **0 = SUPPORT** |
| neutral / **reasoning** (SciNLI) / NEI / NOT ENOUGH INFO / 1 | **1 = NEUTRAL** |
| contradiction / **contrasting** (SciNLI) / REFUTES / 2 | **2 = CONTRADICT** |

Bilinmeyen veya `-1` (SNLI/MNLI gold yok) satırlar atılır. SciNLI'nın "reasoning" sınıfı NEUTRAL'a haritalanır (yumuşak ilişki).

## MODEL

- **Base:** `allenai/scibert_scivocab_uncased` (110M param, bilimsel kelime dağarcığı, vocab=31090)
- **Head:** Dropout(0.1) + Linear(768 → 3)
- **Loss:** CrossEntropy
- **Skalar skor:** `score = P(CONTRADICT) + 0.5 · P(NEUTRAL)` ∈ [0, 1]
  - Gerçek etiket skoru: `{SUPPORT: 0.0, NEUTRAL: 0.5, CONTRADICT: 1.0}`
  - R² bu iki sayı arasında hesaplanır (regresyon-benzeri değerlendirme)

```python
class ContradictionDetector(nn.Module):
    def __init__(self, base, n_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, n_labels))

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    # HF Trainer arayüzü
    def gradient_checkpointing_enable(self, **kw):  self.encoder.gradient_checkpointing_enable(**kw)
    def gradient_checkpointing_disable(self):       self.encoder.gradient_checkpointing_disable()
```

## KONFIGÜRASYON

```python
CONFIG = {
    "base_model": "./models/scibert",      # yerel kopya (Fortinet/AV HF bloklarsa)
    "max_length": 256,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "fp16": True,
    "gradient_checkpointing": False,       # 16GB VRAM rahat alır; ~2x hız
    "output_dir": "./models/contradiction_model",
    "data_dir": "./data",
    "cache_dir": "./cache",
    "hf_cache": "./cache/hf",
}
```

## NOTEBOOK YAPISI (8 HÜCRE)

1. **Setup** — `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP DLL çakışmasını önler), torch'u numpy/pandas'tan ÖNCE import, seed, GPU kontrolü.
2. **Config** — CONFIG dict, klasör oluşturma, `HF_HUB_OFFLINE=1`, ID2LABEL/LABEL2ID.
3. **Birleşik korpus** — SciNLI + FEVER-NLI + MultiNLI indir, normalize, `combined_nli.parquet` cache.
4. **EDA + alt-örnekleme** — Dağılım grafikleri; SciNLI+FEVER tam, MNLI stratified 250k → train_df ~559k, val 5k.
5. **Tokenization** — SciBERT tokenizer (`[CLS] premise [SEP] hypothesis [SEP]`, max_length=256), HF Arrow cache.
6. **Model** — `ContradictionDetector` + parametre sayımı.
7. **Training** — HF Trainer, fp16, eval/save her 2000 step, loss/F1 grafikleri.
8. **SciFact test + demo + kayıt** — Claim+evidence çiftleri 3-sınıfa eşle, R²/Acc/F1, confusion matrix, `detect_contradiction()` fonksiyonu, model dump.

## SciFact TEST PİPELİNE'I

```python
# claims (validation split) + corpus (abstract'lar) -> (claim, abstract, label) çiftleri
SCIFACT_LABEL = {"SUPPORT": 0, "NEI": 1, "NOINFO": 1, "CONTRADICT": 2}
# ev = ex["evidence"]; doc_id -> abstract lookup; label annotator etiketi
# evidence yoksa cited_doc_ids ile NEI olarak ekle
```

## EĞİTİM SÜRESİ

| Konfig | Tahmin |
|---|---|
| ~559k train, 2 epoch, bs=32, gc=False | **~2 saat** (RTX 5080) |
| Eval (5k val) her 2000 step, ~17 eval | ~5 dk toplam |
| SciFact test + kayıt | ~2 dk |
| **TOPLAM** | **~2-2.5 saat** |

İkinci çalıştırmada cache'ler hit eder, eğitim öncesi adımlar ~30 sn.

## HEDEFLER (SciFact test)

| Metrik | Minimum | İdeal |
|---|---|---|
| R² | > 0.55 | > 0.75 |
| Accuracy | > 0.68 | > 0.80 |
| F1 (macro) | > 0.62 | > 0.74 |

> SciNLI'nin train setine dahil edilmesi SciFact transfer performansını belirgin artırır (~3-5 puan F1).

## YARDIMCI SCRIPT'LER

- `scripts/download_data.py` — SciNLI, MNLI, FEVER-NLI, SciFact (claims+corpus) HF'ten indirir, `./cache/hf/` altına.
- `scripts/download_model.py` — SciBERT'i doğrudan URL'den (`config.json`, `vocab.txt`, `pytorch_model.bin`) indirir, `./models/scibert/` altına. SSL verify=False (Fortinet/AV bypass için).

## BİLİNEN SORUNLAR (Windows özel)

| Sorun | Çözüm |
|---|---|
| `c10.dll` WinError 1114 | numpy/pandas'tan ÖNCE `torch` import + `KMP_DUPLICATE_LIB_OK=TRUE` |
| Torch DLL fail | MS VC++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Blackwell (RTX 5080) | `torch --index-url https://download.pytorch.org/whl/cu128` |
| `SSLCertVerificationError` | AV/Fortinet TLS interception → ağ değiştir, sonra offline mode |
| Fortinet "Web Filter Violation" 403 | Aynı (mobile hotspot / WARP) |
| `SciFact: trust_remote_code` | `load_dataset(..., trust_remote_code=True)` |
| `gradient_checkpointing_enable` AttributeError | Custom `nn.Module`'de metodu encoder'a delege et |

## KOD KALİTE STANDARTLARI

- Type hints aktif
- `tqdm` uzun döngülerde
- Cache: parquet (veri) + HF Arrow (tokenized)
- `seed=42` her yerde (random / numpy / torch)
- `logging.INFO` üzerinden log

## VRAM Yönetimi

- `fp16=True` zorunlu
- `gradient_checkpointing=False` (bs=32, seq=256, SciBERT-base → ~6 GB VRAM kullanır, 16GB rahat)
- OOM olursa: bs=24'e düş veya gc=True yap (yavaşlar ama VRAM yarıya iner)

## TESLİM EDİLECEKLER

1. `contradiction_detection.ipynb` — Ana notebook (8 hücre)
2. `scripts/download_data.py` — Veri pre-download
3. `scripts/download_model.py` — Model pre-download (SSL bypass'lı)
4. `requirements.txt` — Pinned dependencies
5. `README.md` — Tek paragraf çalıştırma talimatı + ders açıklaması
6. `models/contradiction_model/` — Final fine-tuned model + metrics.json

# Brain MRI Tumor vs No-Tumor — OOD Generalization (10 Models)

Bu repo, beyin MRI görüntülerinden **Tumor / No Tumor** ikili sınıflandırma problemi için yaptığım (ders projesi kapsamlı) çalışmayı, **gerçek-hayat benzeri çözünürlük dağılımına sahip bir dış test (OOD)** senaryosunda raporlar.

> ⚠️ **Tıbbi kullanım için değildir.** Bu çalışma eğitim/araştırma amaçlıdır.

---

## Neyi farklı yaptım?

Eğitim ve test koşullarını bilerek “aynı dünya” olmaktan çıkardım:

- **Train:** 11.500 görüntü (**256px & 512px sabit** çözünürlük havuzu)
- **Test (External / OOD):** 3.500 görüntü (**190px – 800px değişken** çözünürlük)

Amaç: Modelin sabit input’a “resize” edilmesine rağmen **farklı kaynak + farklı çözünürlük** koşullarında **genelleme** performansını görmek.

Ek olarak raporda tartıştığım bir konu: orijinal veri setlerinde aynı kişiye ait benzer görüntülerin farklı split’lere düşmesi (subject leakage) metrikleri şişirebilir; bu yüzden değerlendirmeyi **dış kaynak test setleri** ile yaptım.

---

## Modeller

Bu zip’te yer alan çıktı/defterlerde görünen deneyler:

- Standart backbone’lar (timm / transfer learning): ConvNeXt-Tiny, DenseNet121, EfficientNet-B0, InceptionV3, MobileNetV2, ResNet34, ResNet50
- Hibrit (ensemble / birleştirme):  
  - DenseNet121 + EfficientNetB0  
  - Swin-T + EfficientNetB0
- Özgün mimari: **My Model (MSAF + EfficientNetB0 tabanlı)**

---

## Sonuç Özeti (OOD test)

Metrikler `results/metrics_summary.csv` dosyasından derlendi.

| experiment | accuracy | auc | f1 | recall_sensitivity | precision | kappa |
| --- | --- | --- | --- | --- | --- | --- |
| My Model / aug (p=0.3) | 0.908 | 0.988 | 0.901 | 0.822 | 0.998 | 0.817 |
| Hybrid - DenseNet121 + EfficientNetB0 / aug (p=0.3) | 0.861 | 0.967 | 0.841 | 0.726 | 1.000 | 0.723 |
| Hybrid - DenseNet121 + EfficientNetB0 / no-aug | 0.839 | 0.939 | 0.812 | 0.684 | 1.000 | 0.680 |
| My Model / no-aug | 0.805 | 0.936 | 0.764 | 0.618 | 0.999 | 0.613 |
| Hybrid 2- SwinEff / aug (p=0.3) | 0.795 | 0.975 | 0.748 | 0.599 | 0.997 | 0.593 |
| resnet34 / no-aug | 0.794 | 0.954 | 0.747 | 0.596 | 0.999 | 0.591 |
| densenet121 | 0.785 | 0.984 | 0.732 | 0.578 | 1.000 | 0.573 |
| convnext_tiny | 0.775 | 0.960 | 0.716 | 0.557 | 1.000 | 0.553 |
| Hybrid 2- SwinEff / no-aug | 0.745 | 0.956 | 0.665 | 0.498 | 1.000 | 0.494 |
| resnet50 / no-aug | 0.719 | 0.962 | 0.619 | 0.448 | 1.000 | 0.444 |
| inception_v3 / no-aug | 0.710 | 0.901 | 0.602 | 0.430 | 1.000 | 0.426 |
| efficientnet_b0 | 0.693 | 0.903 | 0.568 | 0.397 | 0.997 | 0.392 |
| mobilenetv2_100 / no-aug | 0.639 | 0.889 | 0.450 | 0.290 | 1.000 | 0.286 |


> En iyi deney: **My Model / aug (p=0.3)** — Accuracy **0.908**, AUC **0.988**, F1 **0.901**

---

## Hızlı Başlangıç: Tek görselde inference

### 1) Kurulum
```bash
pip install -r requirements.txt
```

### 2) Model ağırlıkları
Ağırlıklar için bağlantı: `reports/model_weights_link.txt`  
(Repo şişmesin diye ağırlıkları Git’e koymadım.)

### 3) Çalıştırma
```bash
python scripts/universal_infer.py --model "PATH/TO/model.pt" --image "PATH/TO/image.jpg" --device cpu
# veya
python scripts/universal_infer.py --model "PATH/TO/model.pt" --image "PATH/TO/image.jpg" --device cuda
```

Detaylar: `scripts/universal_infer_README.txt`

---

## Repo yapısı

- `notebooks/` → eğitim & deney defterleri (ham çalışma akışı)
- `scripts/` → inference script’i ve kullanım dökümanı
- `results/` → metrik özetleri ve model başına raporlar (CSV/TXT)
- `reports/` → proje raporu (docx) + dataset / weight linkleri
- `src/` → ileride kodu “paket” yapmak için iskelet (opsiyonel)

---

## Veri setleri

- Train ana kaynak: Mendeley (rapor içinde link var)
- External test (OOD): Kaggle karışımı (linkler `reports/dataset_links.txt`)

---

## Alıntılama

Bu repo kökünde `CITATION.cff` var. GitHub bunu otomatik “Cite this repository” olarak gösterir.

---

## Lisans

MIT (bkz. `LICENSE`)

# GitHub Yayınlama Rehberi (bu repo için)

Bu dosya, projeyi “portföy + tekrar üretilebilirlik” standardında yayınlamak için pratik bir checklist sunar.

## 1) Repo’yu temizle: “kod var, veri yok”
- Dataset’leri **repo içine koyma** (GitHub boyut limitleri + lisans riskleri).
- `data/`, `datasets/`, `models/`, `checkpoints/` gibi klasörleri `.gitignore` ile dışarıda tut.
- Ağırlıkları (model .pt) için seçenekler:
  - GitHub Releases (küçük dosyalar)
  - Git LFS (büyük dosyalar)
  - Hugging Face Hub / Google Drive (bu repo şu an Drive linki kullanıyor)

## 2) README: ilk 30 saniyeyi kazan
README’de şunlar net olmalı:
- Problem tanımı (Tumor vs No Tumor)
- Train/Test farkı (özellikle OOD çözünürlük farkı)
- “Nasıl çalıştırırım?” (2-3 komut)
- Sonuç tablosu (en iyi model vurgusu)
- Disclaimer: tıbbi kullanım değil

## 3) Reproducibility: minimum standart
- `requirements.txt` veya `environment.yml`
- Sabit random seed’ler (notebook’larda mümkünse)
- Çalışma dizini düzeni (notebooks/results/reports)

## 4) Bilimsel ciddiyet: Model Card + Dataset notu
- `CITATION.cff` ekli.
- Dataset tarafında: kaynak linkleri + lisans/izin uyarıları
- “Model Cards” yaklaşımı: intended use, limitations, metrics, data shift notu

## 5) Sürümleme (release)
- `v1.0.0` etiketi aç
- Release notlarına:
  - en iyi modelin metrikleri
  - ağırlık linki / checksum
  - test senaryosu açıklaması

## 6) CV/LinkedIn için repo vitrini
- Repo açıklamasını şu tarz yap:
  “10 model karşılaştırması + OOD (190–800px) external test ile genelleme analizi”
- Top 3 görsel ekle (ROC curve, confusion matrix, örnek tahmin)


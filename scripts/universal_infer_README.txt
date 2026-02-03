README - Universal Inference (universal_infer.py)
=================================================

Amaç
-----
Bu script, elindeki herhangi bir PyTorch ".pt" modeli (hibrit / özgün / normal timm modelleri)
ve tek bir görüntü dosyası verildiğinde, görüntüyü "tumor" / "no_tumor" olarak sınıflandırmak için yazıldı.

Dosyalar
--------
- universal_infer.py  : İnferans (tahmin) script’i
- Model dosyası (.pt)  : Eğitimden kaydettiğin model/ckpt dosyası
- Görüntü dosyası      : .jpg / .png vb.

Gereksinimler
-------------
Python 3.9+ önerilir.

Paketler:
- torch
- torchvision
- timm
- pillow

Kurulum (pip)
-------------
Windows / Linux / macOS (genel):
    pip install torch torchvision timm pillow

Not:
- CUDA kullanacaksan, sistemine uygun PyTorch sürümünü PyTorch resmi sitesinden kurman gerekebilir.

Çalıştırma
----------
Temel kullanım (model yolu + görüntü yolu):
    python universal_infer.py --model "PATH/TO/model.pt" --image "PATH/TO/photo.jpg"

Cihaz seçimi:
    python universal_infer.py --model "model.pt" --image "photo.jpg" --device cuda
    python universal_infer.py --model "model.pt" --image "photo.jpg" --device cpu

Threshold (eşik) override:
    python universal_infer.py --model "model.pt" --image "photo.jpg" --threshold 0.42

Input size override (varsayılan 256):
    python universal_infer.py --model "model.pt" --image "photo.jpg" --input_size 256

Çıktı
-----
Script iki bölüm basar:

1) MODEL META:
   - Model hangi formatta yüklendi (torchscript / checkpoint / state_dict)
   - Eğer bulunursa model_name, input_size, mean/std, threshold vb.

2) PREDICTION:
   - pred_label  : tahmin etiketi (no_tumor veya tumor)
   - p_tumor     : tümör olasılığı
   - p_no_tumor  : tümör değil olasılığı
   - threshold   : kullanılan eşik

Önemli Notlar (Model Dosyası Formatı)
-------------------------------------
Bu script şu sırayla yüklemeyi dener:

A) TorchScript (.pt):
   - Eğer modelin torch.jit.save ile kaydedildiyse, en sorunsuz biçim budur.
   - Script direkt torch.jit.load ile açar.

B) Checkpoint / state_dict (.pt):
   - torch.save({ ... }) formatında bir dict olabilir (checkpoint)
   - veya doğrudan state_dict olabilir

En garanti yöntem:
- checkpoint içine şu alanları kaydet:
    model_name, state_dict, input_size, mean, std, class_names, best_threshold

Eğer metadata yoksa:
- Script state_dict anahtarlarına bakıp (hibrit/özgün) tahmin eder,
- ya da bilinen aday modelleri deneyip en iyi eşleşeni seçer.
Bu çoğu zaman çalışır ama “%100 garanti” için metadata önerilir.

Sık Hatalar ve Çözümler
-----------------------
1) "ModuleNotFoundError: No module named 'timm'"
   -> pip install timm

2) "PIL" hatası
   -> pip install pillow

3) CUDA bulunamadı / GPU kullanmıyor
   -> --device cpu ile çalıştır ya da sistemine uygun CUDA PyTorch kur.

4) "State_dict hiçbir bilinen modele uymadı"
   -> Bu model mimarisi listede yok demektir.
      Çözüm: eğitimde kullandığın model_name’i checkpoint’e kaydet veya
      script içine o mimarinin build_model karşılığını ekle.

Windows’ta Yol Yazımı
---------------------
Örnek:
    python universal_infer.py --model "C:\\proj\\runs\\best.pt" --image "C:\\proj\\test\\img.jpg"

Güvenli yol yazımı için ters slash yerine:
    python universal_infer.py --model "C:/proj/runs/best.pt" --image "C:/proj/test/img.jpg"

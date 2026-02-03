import argparse
import os
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms


# -------------------------
# Preprocess yardımcıları
# -------------------------
class SquarePad:
    """Dikdörtgen görüntüyü kareye tamamlamak için padding."""
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        m = max(w, h)
        pad_left = (m - w) // 2
        pad_top = (m - h) // 2
        pad_right = m - w - pad_left
        pad_bottom = m - h - pad_top
        # siyah padding
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def build_transform(input_size: int, mean: List[float], std: List[float]):
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------------------------
# Model tanımları (notebook ile uyumlu)
# -------------------------
class HybridDN121_EffB0(nn.Module):
    """
    DenseNet121 + EfficientNetB0 hibrit:
    - İki backbone pooled feature (num_classes=0, global_pool='avg')
    - concat + küçük MLP head
    """
    def __init__(self, num_classes=2, head_dim=256, dropout=0.2, freeze_backbones=False):
        super().__init__()
        self.bb1 = timm.create_model("densenet121", pretrained=True, num_classes=0, global_pool="avg")
        self.bb2 = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")

        if freeze_backbones:
            for p in self.bb1.parameters(): p.requires_grad = False
            for p in self.bb2.parameters(): p.requires_grad = False

        f1 = self.bb1.num_features
        f2 = self.bb2.num_features
        self.head = nn.Sequential(
            nn.Linear(f1 + f2, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes)
        )

    def forward(self, x):
        v1 = self.bb1(x)
        v2 = self.bb2(x)
        v = torch.cat([v1, v2], dim=1)
        return self.head(v)


class SEBlock(nn.Module):
    """Squeeze-Excitation: kanal bazlı attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class Conv1x1BNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class CustomMSAF_EffB0(nn.Module):
    """
    Özgün model:
    EfficientNet-B0 features_only -> 3 ölçek feature map
    Her ölçekte 1x1 proj + SE
    Ölçek attention ile weighted fusion + concat(all) -> MLP head
    """
    def __init__(
        self,
        num_classes=2,
        out_indices=(2, 3, 4),
        embed_dim=256,
        head_dim=256,
        dropout=0.35,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=out_indices
        )
        chs = self.backbone.feature_info.channels()  # list of channels per scale

        self.proj = nn.ModuleList([Conv1x1BNAct(c, embed_dim) for c in chs])
        self.se   = nn.ModuleList([SEBlock(embed_dim, reduction=16) for _ in chs])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        att_hidden = max(8, embed_dim // 4)
        self.scale_attn = nn.Sequential(
            nn.Linear(embed_dim, att_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(att_hidden, 1),
        )

        in_dim = embed_dim * (len(chs) + 1)  # fused + concat(all)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)  # list of (B,C,H,W)
        vecs = []
        for f, proj, se in zip(feats, self.proj, self.se):
            f = se(proj(f))
            v = self.pool(f).flatten(1)  # (B, E)
            vecs.append(v)

        ms = torch.stack(vecs, dim=1)             # (B,S,E)
        att_logits = self.scale_attn(ms).squeeze(-1)  # (B,S)
        w = torch.softmax(att_logits, dim=1)      # (B,S)

        fused = (ms * w.unsqueeze(-1)).sum(dim=1) # (B,E)
        concat_all = ms.flatten(1)                # (B,S*E)
        z = torch.cat([fused, concat_all], dim=1) # (B,(S+1)*E)
        return self.head(z)


# Notebook’taki MODEL_PROFILES (bilinen adaylar)
KNOWN_MODEL_NAMES = [
    "resnet34",
    "resnet50",
    "densenet121",
    "inception_v3",
    "efficientnet_b0",
    "mobilenetv2_100",
    "convnext_tiny",
    "hybrid_dn121_effb0",
    "custom_msaf_effb0",
]


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "hybrid_dn121_effb0":
        return HybridDN121_EffB0(num_classes=num_classes, head_dim=256, dropout=0.2, freeze_backbones=False)
    if model_name == "custom_msaf_effb0":
        return CustomMSAF_EffB0(num_classes=num_classes, embed_dim=256, head_dim=256, dropout=0.35, freeze_backbone=False)
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


# -------------------------
# Checkpoint / model yükleme
# -------------------------
def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(isinstance(k, str) for k in obj.keys())


def _try_load_torchscript(model_path: str, device: torch.device) -> Optional[nn.Module]:
    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        return m
    except Exception:
        return None


def _score_state_dict_match(model: nn.Module, sd: Dict[str, torch.Tensor]) -> Tuple[bool, int]:
    """
    strict=True ile yüklemeyi dene. Olursa perfect match.
    strict=False ile de deneyeceğiz ama önce strict hedef.
    """
    try:
        model.load_state_dict(sd, strict=True)
        return True, 0
    except Exception:
        return False, 10**9


def _best_guess_model_name_from_keys(keys: List[str]) -> Optional[str]:
    # Hibritin state_dict anahtarlarında genelde bb1./bb2. vardır
    if any(k.startswith("bb1.") for k in keys) and any(k.startswith("bb2.") for k in keys):
        return "hybrid_dn121_effb0"
    # Custom modelde scale_attn, proj, se anahtarları olur
    if any(k.startswith("scale_attn.") for k in keys) and any(k.startswith("proj.") for k in keys):
        return "custom_msaf_effb0"
    return None


def load_any_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Dönüş:
      model: eval modunda
      meta: input_size/mean/std/class_names/threshold/model_name gibi bilgiler (varsa)
    """
    # 1) TorchScript dene
    ts = _try_load_torchscript(model_path, device)
    if ts is not None:
        return ts.to(device), {"format": "torchscript"}

    # 2) torch.load dene
    obj = torch.load(model_path, map_location=device)

    # 2a) Direkt model objesi ise
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj.to(device), {"format": "torch_model_object"}

    meta: Dict[str, Any] = {"format": "checkpoint_or_state_dict"}

    # 2b) checkpoint dict ise
    state_dict = None
    if isinstance(obj, dict):
        # bazı checkpointlerde "state_dict" yerine direkt model ağırlıkları olabilir
        if "state_dict" in obj and _looks_like_state_dict(obj["state_dict"]):
            state_dict = obj["state_dict"]
            # meta anahtarları
            for k in ["model_name", "input_size", "mean", "std", "class_names", "threshold", "best_threshold"]:
                if k in obj:
                    meta[k] = obj[k]
        elif _looks_like_state_dict(obj):
            state_dict = obj
        elif "model" in obj and isinstance(obj["model"], nn.Module):
            m = obj["model"]
            m.eval()
            return m.to(device), {"format": "checkpoint_has_model_object"}

    if state_dict is None:
        raise RuntimeError("Model dosyası TorchScript değil ve state_dict/nn.Module da değil. Format desteklenmiyor.")

    # DataParallel ise module. prefix temizle
    state_dict = _strip_prefix(state_dict, "module.")

    # 3) model_name metadan varsa kullan
    model_name = meta.get("model_name", None)
    if model_name is None:
        # 4) anahtar heuristiği
        guess = _best_guess_model_name_from_keys(list(state_dict.keys()))
        if guess is not None:
            model_name = guess

    # 5) model_name hala yoksa: bilinen adayları brute-force dene (en güvenlisi)
    candidates = KNOWN_MODEL_NAMES if model_name is None else [model_name]

    best_name = None
    best_model = None
    best_score = 10**9

    for name in candidates:
        try:
            m = build_model(name, num_classes=2).to(device)
            ok, score = _score_state_dict_match(m, state_dict)
            if ok:
                best_name = name
                best_model = m
                best_score = 0
                break
        except Exception:
            continue

    if best_model is None:
        # strict=True hiçbiri olmadıysa strict=False ile en iyi yaklaşımı seç
        for name in KNOWN_MODEL_NAMES:
            try:
                m = build_model(name, num_classes=2).to(device)
                missing, unexpected = m.load_state_dict(state_dict, strict=False)
                score = len(missing) + len(unexpected)
                if score < best_score:
                    best_score = score
                    best_name = name
                    best_model = m
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("State_dict hiçbir bilinen modele uymadı. Eğitimde farklı mimari/format kullanılmış olabilir.")

    best_model.eval()
    meta["model_name"] = best_name
    meta["match_score"] = best_score
    return best_model, meta


# -------------------------
# Inference
# -------------------------
def predict_one(model: nn.Module, img_path: str, device: torch.device,
                input_size: int, mean: List[float], std: List[float],
                class_names: List[str], threshold: float) -> Dict[str, Any]:

    img = Image.open(img_path).convert("RGB")
    tfm = build_transform(input_size=input_size, mean=mean, std=std)
    x = tfm(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)

        # Çıkış şekli (B,2) bekliyoruz
        if logits.ndim == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[0]
            p0 = float(probs[0].item())
            p1 = float(probs[1].item())
            pred_idx = 1 if p1 >= threshold else 0
            return {
                "pred_idx": pred_idx,
                "pred_label": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
                "p_no_tumor": p0,
                "p_tumor": p1,
                "threshold": threshold,
            }

        # Tek logit (B,1) ya da (B,) olursa sigmoid ile yorumla
        if logits.ndim == 2 and logits.shape[1] == 1:
            p1 = float(torch.sigmoid(logits)[0, 0].item())
        elif logits.ndim == 1:
            p1 = float(torch.sigmoid(logits)[0].item())
        else:
            raise RuntimeError(f"Beklenmeyen logits shape: {tuple(logits.shape)}")

        pred_idx = 1 if p1 >= threshold else 0
        return {
            "pred_idx": pred_idx,
            "pred_label": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
            "p_tumor": p1,
            "p_no_tumor": 1.0 - p1,
            "threshold": threshold,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model .pt yolu")
    ap.add_argument("--image", required=True, help="Tahmin edilecek görüntü yolu")
    ap.add_argument("--device", default=None, help="cuda / cpu (boşsa otomatik)")
    ap.add_argument("--threshold", type=float, default=None, help="Varsa threshold (boşsa 0.5 veya checkpoint'ten)")
    ap.add_argument("--input_size", type=int, default=None, help="Varsa input size override (boşsa checkpoint/256)")
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, meta = load_any_model(args.model, device=device)

    # input_size / mean / std
    input_size = 256
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    if isinstance(meta.get("input_size", None), int):
        input_size = int(meta["input_size"])
    if isinstance(meta.get("mean", None), (list, tuple)) and len(meta["mean"]) == 3:
        mean = list(map(float, meta["mean"]))
    if isinstance(meta.get("std", None), (list, tuple)) and len(meta["std"]) == 3:
        std = list(map(float, meta["std"]))

    if args.input_size is not None:
        input_size = args.input_size

    # class names
    class_names = meta.get("class_names", None)
    if not (isinstance(class_names, (list, tuple)) and len(class_names) >= 2):
        # default: ImageFolder’da genelde "no_tumor" ve "tumor"
        class_names = ["no_tumor", "tumor"]

    # threshold
    threshold = 0.5
    if meta.get("best_threshold", None) is not None:
        try:
            threshold = float(meta["best_threshold"])
        except Exception:
            pass
    if meta.get("threshold", None) is not None:
        try:
            threshold = float(meta["threshold"])
        except Exception:
            pass
    if args.threshold is not None:
        threshold = args.threshold

    out = predict_one(
        model=model,
        img_path=args.image,
        device=device,
        input_size=input_size,
        mean=mean,
        std=std,
        class_names=list(class_names),
        threshold=threshold,
    )

    print("=== MODEL META ===")
    for k in sorted(meta.keys()):
        print(f"{k}: {meta[k]}")
    print("\n=== PREDICTION ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

"""
mouth_patch_export.py（2026/11/24；v6 口鼻小塊模型的第 4 步，在 .venv-vlm 跑：要 PyTorch、torchvision、onnx）
三顆口鼻小塊模型（mouth_patch_train.py 的 model_patch_<tag>_s0..2.pt）合成一個 ONNX，主流程用 onnxruntime 跑（不用 PyTorch）。
輸入 bgr：N×96×96×3 uint8（OpenCV 的 BGR，就是 infant_face_v6.mouth_patch 切出來的樣子）；
圖裡面做：轉 float、/255、BGR→RGB、ImageNet 正規化、三顆各算 sigmoid 再平均。
輸出 p_visible：N（口鼻露出的機率，三顆平均）。

    .venv-vlm/Scripts/python -m experiments.mouth_patch_export p_base model/mouth_patch_ens3.onnx
輸出：model/mouth_patch_ens3.onnx（134 MB，不進 repo）

--hailo：另外匯出給 Hailo DFC 編 HEF 的版本（hailo/compile_hef.py --model mouth）：
  輸入 bgr_norm：1×3×96×96 float，已經正規化的 BGR（正規化放在 HEF 的 model script 裡做，晶片直接吃 uint8 小塊）；
  BGR→RGB 折進三顆第一層卷積的輸入通道順序（數學上一樣，晶片上不用翻通道）；
  輸出 logits：1×3（三顆各自的 logit；sigmoid、平均在主機做，hailo/mouth_hef.py）。

    .venv-vlm/Scripts/python -m experiments.mouth_patch_export p_base model/mouth_patch_ens3_hailo.onnx --hailo
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "outputs" / "airway3" / "patch"


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, bgr):
        x = bgr.float().div(255.0).permute(0, 3, 1, 2).flip(1)       # N×3×H×W，RGB
        x = (x - self.mean) / self.std
        return torch.stack([torch.sigmoid(m(x).squeeze(1)) for m in self.models], 1).mean(1)


class HailoEnsemble(nn.Module):
    """給 DFC 的版本：吃正規化過的 BGR，吐三顆的 logit（見檔頭 --hailo）。"""

    def __init__(self, models):
        super().__init__()
        models = [copy.deepcopy(m) for m in models]
        for m in models:   # 第一層卷積的輸入通道 RGB → BGR：吃 BGR 的結果跟原本吃 RGB 的一樣
            m.conv1.weight.data = m.conv1.weight.data[:, [2, 1, 0]].clone()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.cat([m(x) for m in self.models], 1)


def main():
    args = [a for a in sys.argv[1:] if a != "--hailo"]
    if len(args) != 2:
        raise SystemExit("用法：python -m experiments.mouth_patch_export <tag，例如 p_base> <輸出 .onnx> [--hailo]")
    tag, out = args[0], ROOT / args[1]
    models = []
    for s in range(3):
        m = torchvision.models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, 1)
        m.load_state_dict(torch.load(P / f"model_patch_{tag}_s{s}.pt", map_location="cpu"))
        models.append(m.eval())
    out.parent.mkdir(parents=True, exist_ok=True)
    if "--hailo" in sys.argv:
        torch.onnx.export(HailoEnsemble(models).eval(), (torch.zeros(1, 3, 96, 96),), str(out),
                          input_names=["bgr_norm"], output_names=["logits"], opset_version=13, dynamo=False)
    else:
        dummy = torch.zeros(2, 96, 96, 3, dtype=torch.uint8)
        torch.onnx.export(Ensemble(models).eval(), (dummy,), str(out), input_names=["bgr"], output_names=["p_visible"],
                          dynamic_axes={"bgr": {0: "n"}, "p_visible": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"→ {out}（{out.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()

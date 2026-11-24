# Third-party notices

This repository redistributes one third-party file: the YuNet face detection model. Everything else listed below is
downloaded at install or run time and is not included here.

## Included in this repository

### YuNet face detector — `model/face_detection_yunet_2023mar.onnx`

Source: OpenCV Zoo, https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Used but not included

| Component | Used for | License |
| --- | --- | --- |
| RTMO-m, YOLOX-m, RTMPose-m (OpenMMLab / MMPose ONNX weights) | Pose estimation; downloaded by rtmlib on first run | Apache-2.0 |
| [rtmlib](https://github.com/Tau-J/rtmlib) | Runs the pose models | Apache-2.0 |
| OpenCV (`opencv-python`) | Image and video I/O, YuNet inference | Apache-2.0 (OpenCV), MIT (packaging) |
| ONNX Runtime | Inference backend for rtmlib and for the mouth/nose patch model | MIT |
| [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k) pretrained weights | Starting point of the v6 mouth/nose patch model (`experiments/mouth_patch_train.py`); the trained model `model/mouth_patch_ens3.onnx` is not in this repository | Apache-2.0 (model card) |
| PyTorch, torchvision, safetensors, onnx | Training and exporting the mouth/nose patch model (separate `.venv-vlm` environment); the pipeline runs the exported ONNX file without them | BSD-3-Clause (PyTorch, torchvision), Apache-2.0 (safetensors, onnx) |
| Pillow | Image handling | MIT-CMU |
| Qwen2-VL-2B-Instruct | Experimental VLM second check (`experiments/vlm_spike.py`, separate environment) | Apache-2.0 |
| torchvision's ImageNet-pretrained ResNet18 weights | Earlier head-crop classifier experiment (`experiments/airway_train.py`); not part of the pipeline | PyTorch and torchvision: BSD-3-Clause. The weights have no separate licence; torchvision notes that pretrained weights may be subject to the terms of their training data (ImageNet allows non-commercial use only). Whether a model fine-tuned from them can ship is still open |
| Ultralytics YOLOv8 (optional `yolo` dependency group) | Only the early v1/v2 scripts and the YOLOv8 comparison in `evaluation/eval_syrip.py`; not part of the pipeline | AGPL-3.0 |

Compiled Hailo files (`.hef`, `.har`) are not included; see [`hailo/結果.md`](hailo/結果.md).

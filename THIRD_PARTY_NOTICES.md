# Third-party notices

This repository redistributes one third-party file: the YuNet face detection model. Everything else listed below is
not included here: libraries and models are installed or downloaded separately (at install, on first run, or by hand
as the README describes), and the last two sections cover our own trained weights and the data we used.

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
| RTMO-m, YOLOX-m, RTMPose-m ([OpenMMLab MMPose](https://github.com/open-mmlab/mmpose), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo); ONNX weights) | Pose estimation; downloaded by rtmlib on first run. Our quantization-aware retrained RTMO-m and its HEF (`hailo/qat_rtmo.py`) start from these weights and are not in this repository | Apache-2.0 |
| [rtmlib](https://github.com/Tau-J/rtmlib) | Runs the pose models | Apache-2.0 |
| [OpenCV](https://github.com/opencv/opencv) ([`opencv-python`](https://github.com/opencv/opencv-python)) | Image and video I/O, YuNet inference | Apache-2.0 (OpenCV), MIT (packaging) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Inference backend for rtmlib, the mouth/nose patch model and RetinaFace | MIT |
| [NumPy](https://github.com/numpy/numpy) | Array maths throughout | BSD-3-Clause |
| [psutil](https://github.com/giampaolo/psutil) | CPU load in the benchmarks | BSD-3-Clause |
| RetinaFace MobileNet v1 (`retinaface_mobilenet_v1` in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo), [network config](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/retinaface_mobilenet_v1.yaml); architecture from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)) | v7's second face detector, run only when v6 finds no uncovered face in a crib (`pipeline/infant_face_v7.py`). Downloaded as Hailo's TFLite file; `hailo/retina_port.py` copies its weights unchanged into a 640×480 ONNX file and `hailo/compile_hef.py --model retina` compiles that for the UGen300. Neither file is in this repository. The config lists WIDER FACE as the training data, as for YuNet | MIT (as stated in Hailo's network config) |
| [`tflite`](https://github.com/zhenhuaw-me/tflite), [LiteRT](https://github.com/google-ai-edge/LiteRT) (`ai-edge-litert`) | Reading the RetinaFace TFLite file for the port, and checking the port against it (`hailo/retina_port.py`); not used by the pipeline | Apache-2.0 |
| [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k) pretrained weights | Starting point of the v6 mouth/nose patch model (`experiments/mouth_patch_train.py`); the trained model `model/mouth_patch_ens3.onnx` is not in this repository | Apache-2.0 (model card) |
| [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision), [safetensors](https://github.com/huggingface/safetensors), [onnx](https://github.com/onnx/onnx) | Training and exporting the mouth/nose patch model, and porting RetinaFace to ONNX (separate `.venv-vlm` environment); the pipeline runs the exported ONNX files without them | BSD-3-Clause (PyTorch, torchvision), Apache-2.0 (safetensors, onnx) |
| [Pillow](https://github.com/python-pillow/Pillow) | Image handling | MIT-CMU |
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Experimental VLM second check (`experiments/vlm_spike.py`; on the UGen300 through Hailo's precompiled HEF from the [GenAI Model Zoo](https://github.com/hailo-ai/hailo_model_zoo_genai), `hailo/vlm_hef.py`); not part of the pipeline, and the HEF is not included | Apache-2.0 (model weights) |
| torchvision's ImageNet-pretrained ResNet18 weights | Earlier head-crop classifier experiment (`experiments/airway_train.py`) only; nothing fine-tuned from them is part of the submitted work | The weights have no separate licence; torchvision notes that pretrained weights may be subject to the terms of their training data (ImageNet allows non-commercial use only). Following the organizer's written reply (2026-11-24), the pipeline's mouth/nose model starts from weights labelled Apache-2.0 instead (`timm/resnet18.a1_in1k` above) |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (optional `yolo` dependency group) | Only the early v1/v2 scripts and the YOLOv8 comparison in `evaluation/eval_syrip.py`; not part of the pipeline | AGPL-3.0 |

Compiled Hailo files (`.hef`, `.har`) are not included; see [`hailo/結果.md`](hailo/結果.md).

## Our trained weights

Not in this repository (like the other weights). Both start from Apache-2.0 base weights, which allow modification
and sublicensing, and are otherwise trained only on our own doll footage and, for the pose model, COCO images with
licences that allow commercial use:

- Quantization-aware retrained RTMO-m (`hailo/qat_rtmo.py`) and its HEF: base RTMO-m (Apache-2.0).
- Mouth/nose patch model `model/mouth_patch_ens3.onnx` (`experiments/mouth_patch_*.py`): base `timm/resnet18.a1_in1k` (Apache-2.0).

## Data that is not part of the work

- SyRIP (real infant photos; Augmented Cognition Lab, Northeastern University): used only to measure false alerts, with
  the authors' permission for this competition; never used for training, and no image or output image is in this
  repository, the slides or the video. Results quoted from it give the number of images.
- [COCO 2017](https://cocodataset.org/) images with licences that allow commercial use (CC BY and similar; val for HEF
  calibration, train for the pose model's quantization-aware training): used only there, not redistributed.

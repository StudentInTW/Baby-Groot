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
| ONNX Runtime | Inference backend for rtmlib | MIT |
| Pillow | Image handling | MIT-CMU |
| Qwen2-VL-2B-Instruct | Experimental VLM second check (`vlm_spike.py`, separate environment) | Apache-2.0 |
| Ultralytics YOLOv8 (optional `yolo` dependency group) | Only the early v1/v2 scripts and the YOLOv8 comparison in `eval_syrip.py`; not part of the v3 pipeline | AGPL-3.0 |

Compiled Hailo files (`.hef`, `.har`) are not included; see [`hailo/結果.md`](hailo/結果.md).

# Motorbike Helmet Detection — real-time web app

Final project, Advanced Deep Learning (Phenikaa University) — Dang Quang Anh, Vu Tuan Minh.

Open the page on a phone, press **Start camera** and point it at the street. The app detects **motorbikes**, **persons**
and **helmets** and marks every motorbike as **HELMET OK** (green), **NO HELMET** (red) or **?** (rider too small / unclear).

* Model: YOLO11n fine-tuned from COCO weights on our 383-image Vietnamese traffic dataset, exported to ONNX
  (`model_640.onnx` = fast mode, `model_960.onnx` = accurate mode for small, distant helmets).
* Runs **entirely on the device** with ONNX Runtime Web (WebGPU, or WebAssembly as a fallback). No image leaves the phone.
* Helmet status: a person overlapping a motorbike is a rider; a helmet box in the rider's head region means *helmet*;
  a rider at least 80 px tall without one means *no helmet*. Motorbikes are tracked between frames to count unique violators.
* Controls: switch front/back camera, fast/accurate mode, confidence threshold, or test with a photo/video file.

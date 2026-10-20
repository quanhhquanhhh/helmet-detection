# Motorbike Helmet Detection — real-time web app

Final project, Advanced Deep Learning (Phenikaa University) — Dang Quang Anh, Vu Tuan Minh.

Open the page on a phone, press **Start camera** and point it at the street. The app detects **motorbikes**, **persons**
and **helmets** and marks every motorbike as **HELMET OK** (green), **NO HELMET** (red) or **?** (rider too small / unclear).

* Model: YOLO26n fine-tuned from COCO weights on our Vietnamese traffic dataset + 191 street-level photos, exported to ONNX
  (one dynamic-size model, split into `model_part1.bin` + `model_part2.bin`).
* Runs **entirely on the device** with ONNX Runtime Web (WebGPU, or WebAssembly as a fallback). No image leaves the phone.
* Helmet status: a person overlapping a motorbike is a rider; a helmet box in the rider's head region means *helmet*;
  a rider at least 80 px tall without one means *no helmet*. Motorbikes are tracked between frames to count unique violators.
* Fully automatic: multi-scale inference (960 px for small, distant objects + 320 px for large, close objects), per-class thresholds chosen on the validation set, speed adaptation to the phone, and per-motorbike tracking that smooths the helmet status over several frames.

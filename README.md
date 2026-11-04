# Motorbike Helmet Detection — real-time web app

Final project, Advanced Deep Learning (Phenikaa University) — Dang Quang Anh, Vu Tuan Minh.

Open the page on a phone, press **Start camera** and point it at the street. The app detects **motorbikes**, **persons**,
**helmets**, **bare heads** and marks every motorbike as **HELMET OK** (green), **NO HELMET** (red) or **?** (rider too small / unclear).

* Model: YOLO26n (classes motorbike, person, helmet, no_helmet) fine-tuned from COCO weights on our Vietnamese traffic dataset
  + 191 street-level photos + frames of the HELMET dataset (Myanmar) labelled automatically from its per-motorbike
  annotations + frames of Hanoi street videos (labels checked by eye; the frames are not published), exported to ONNX
  (one dynamic-size model, split into `model_part1.bin` + `model_part2.bin`).
* Runs **entirely on the device** with ONNX Runtime Web (WebGPU, or WebAssembly as a fallback). No image leaves the phone.
* Helmet status: a person overlapping a motorbike is a rider; a helmet box in the rider's head region means *helmet*,
  a no_helmet box means *no helmet*; a rider with neither is *unclear*. Motorbikes are tracked between frames to count unique violators.
* Fully automatic: multi-scale inference (960 px for small, distant objects + 320 px for large, close objects), per-class thresholds chosen on held-out Hanoi video frames, speed adaptation to the phone, and per-motorbike tracking that smooths the helmet status over several frames.

* Vietnamese traffic rules checked (Law 36/2024/QH15 on Road Traffic Order and Safety, Art. 33): the rider and every
  passenger of a motorbike must wear a helmet. Pedestrians, bicycles and parked bikes are not checked. The violation log shows the fines of Decree 168/2024/ND-CP as published.
* Settings (slider button): confidence threshold per class, kept in the browser; Reset returns to the values chosen on
  held-out Hanoi video frames.

HELMET dataset: H. Lin, J. D. Deng, D. Albers, F. W. Siebert, "Helmet Use Detection of Tracked Motorcycles Using CNN-Based
Multi-Task Learning", IEEE Access, 2020; data by F. W. Siebert and H. Lin, https://osf.io/4pwj8, licensed CC BY 4.0.
Only the trained model is published here; no dataset images are included.

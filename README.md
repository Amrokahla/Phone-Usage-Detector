# 📘 Active Phone Usage Detection — Detailed Thinking Flow & Tech Stack

A comprehensive, implementation-ready plan combining **problem framing**, **reasoning/decision logic**, and **technology choices**. Use it as a build guide and checklist. *(No code; all design + notes.)*

---

## 0) Objectives & Success Criteria

**Goal:** Detect **active phone usage** in videos and render bounding boxes + confidence **only** when usage is occurring. Keep original resolution, FPS, and audio.

**Success Criteria**

* Boxes appear **only** during active usage; **no false positives** for static/mounted phones.
* Smooth playback (no flicker); near real-time processing on target hardware.
* Optional: accurate timestamps and clear summary report.

**Out of Scope**

* Other distractions (books, tablets), true real-time streaming, multi-person identity tracking.

---

## 1) Definitions (tighten the spec)

**Active usage** (any of the following):

1. **In-hand:** phone touched/held; finger/thumb micro-motions (scroll/tap) or phone body motion.
2. **Near-face:** phone within a tight distance of face/ear (speaking or reading), sustained for a dwell time.
3. **On-lap + hand contact:** phone low in frame height (lap region) **and** hand proximity + intermittent micro-motion.

**Ignore:**

* Phone static on table/holder/mount without hand proximity.
* Phone visible but **no interaction signals** (no hand, no motion, no near-face dwell).

---

## 2) Key Signals (what evidence we’ll rely on)

**Spatial what/where:**

* **Phone boxes** (object detector).
* **Hand keypoints/boxes** (for proximity + micro-motion of fingertips).
* **Face box/landmarks** (for near-face cue; optional head-pose).

**Temporal dynamics:**

* **Phone motion** (centroid velocity or optical flow inside phone box).
* **Finger micro-motion** (small but frequent fingertip displacements).
* **Dwell time** (proximity sustained across frames).

**Interactions:**

* **Hand–phone proximity** (contact/near-contact threshold).
* **Phone–face proximity** (normalized by face size).
* **Lap region heuristic** (normalized y-position in image).

> **Note:** All distances normalized by a relevant scale (e.g., min(phone\_w, phone\_h) or face\_h) so thresholds generalize across resolutions.

---

## 3) End-to-End Pipeline (with technologies)

### A) Ingest & Prep

* **Read frames** at native FPS/resolution; keep **audio** aside for re-muxing.
* **Processing size:** optionally downscale to **720p** for inference; keep a mapping to render boxes at native size.
* **Technologies:** **FFmpeg** (mux/demux, preserve FPS/audio), **OpenCV**/**PyAV** (frame access & drawing), **MoviePy** (optional edit convenience).

### B) Detection (every K frames)

* **Phone detection:** YOLO-family (fast, small).
* **Hands:** keypoints or tight hand boxes.
* **Face:** robust, lightweight face detector.
* **Technologies:**

  * **Phones:** **YOLOv8n/s** or **YOLO-NAS** (export to **ONNX Runtime** / **TensorRT**).
  * **Hands:** **MMPose** (light backbones: MobileNetV2, Lite-HRNet) or **YOLO-Pose** (hand variant).
  * **Face:** **RetinaFace (mobilenet-0.25)** or **YOLO-face**.
  * **Accel:** **ONNX Runtime** (CPU/GPU), **TensorRT** (NVIDIA), quantize FP16/INT8.

### C) Tracking (every frame)

* Track phones, hands, faces between detections; provide IDs for short windows (no global identity).
* **Technologies:** **ByteTrack** (preferred), **DeepSORT** (if appearance helps), or **NorFair** (lightweight).
* **Notes:** IoU/Kalman gating; re-ID window \~0.5–1.0 s; max\_age \~15–20 frames.

### D) Feature Extraction (sliding window W = 0.5–1.0 s)

For each **phone tracklet**:

* **d\_hand\_phone:** min distance fingertip/wrist ↔ phone bbox (normalized by phone size).
* **phone\_motion:** mean centroid speed or mean optical-flow magnitude inside phone box.
* **d\_phone\_face:** phone ↔ face distance / face\_height.
* **finger\_motion:** variance/mean speed of thumb/index tips.
* **lap\_flag:** phone\_y\_center / frame\_h ≥ τ\_lap (e.g., 0.70–0.80).
* **dwell:** consecutive frames under proximity thresholds.

**Technologies:** **OpenCV** optical flow (Farneback) *or* centroid velocity; keypoints from **MMPose**; boxes from YOLO/RetinaFace; temporal buffers in memory.

### E) Decision Logic (two-stage)

**Stage 1 — Rule Baseline (conservative, tunable):**

* **In-hand:**

  * `d_hand_phone ≤ T1` for ≥ **M\_on** frames **AND** (`phone_motion ≥ Tmot` **OR** `finger_motion ≥ Tfing`).
* **Near-face:**

  * `(d_phone_face / face_h) ≤ T2` for ≥ **M\_on** frames (motion optional).
* **On-lap:**

  * `lap_flag = true` **AND** `d_hand_phone ≤ T3` **AND** (`phone_motion ≥ Tmot_low` **OR** `finger_motion ≥ Tfing_low`) within window.
* **Ignore static:**

  * `phone_motion < Tmot_static` for ≥ **S\_static** frames **AND** `d_hand_phone > T_ignore`.
* **Hysteresis:**

  * Turn **ON** after `M_on` consecutive frames meeting condition; turn **OFF** only after `M_off` frames violating it. (Prevents flicker.)
* **Veto rules:**

  * If phone area / frame area > α (implausibly big) or aspect ratio outside \[0.45, 0.75]\* (portrait phone \~0.5–0.6), **veto** usage unless strong hand contact.

**Stage 2 — ML Upgrade (when data available):**

* Train a tiny **temporal classifier** on features above to output `P(active_usage)`.
* **Tech:** **XGBoost/LightGBM** (tabular) or **PyTorch** TCN/LSTM (sequence). Keep stage-1 rules as guardrails/veto.

> \*Aspect ratio ranges depend on labeling; the note is a practical prior to fight false positives (e.g., books, tablets).

### F) Occlusion Robustness

* **Keep-alive:** maintain ON state across short occlusions ≤ **0.5 s**, if recent features strongly indicated usage.
* **Motion-only grace:** if hands vanish but the phone remains moving (or near-face persists), keep ON for a grace window.
* **Re-association:** use position + short-term velocity to re-bind phone/hand after partial occlusions.
* **Notes:** Prefer conservative re-activation after long occlusions (require full conditions again).

### G) Overlay & Output

* Draw **only when `active_usage = ON`**. Label: `Phone (usage: 0.95)` → probability from rules/ML.
* **Smoothing:** linear interpolate bbox between detection frames; EMA of box corners to remove jitter.
* **Output:** render overlays at native resolution/FPS; **re-mux** original audio untouched.
* **Technologies:** **OpenCV** for drawing; **FFmpeg** for final muxing.

---

## 4) Default Thresholds & Windows (MVP cheatsheet)

*(Tune per camera & hardware; all times at 30 FPS unless noted.)*

| Parameter                |                       Default | Rationale/Notes                       |
| ------------------------ | ----------------------------: | ------------------------------------- |
| Processing size          |                      1280×720 | Good speed/accuracy tradeoff          |
| K (detection interval)   |                      3 frames | Detect at 10 Hz; track in-between     |
| Window **W**             |             15 frames (0.5 s) | Temporal stability for features       |
| `T1` hand–phone          |  0.10×min(phone\_w, phone\_h) | Contact/near-contact                  |
| `T2` phone–face          |                  0.30×face\_h | Near-face threshold                   |
| `T3` on-lap contact      |  0.12×min(phone\_w, phone\_h) | Slightly looser than T1               |
| `τ_lap` lap region       |     y\_center/frame\_h ≥ 0.75 | Cabin/lap heuristic                   |
| `Tmot` phone motion      |        1.2 px/frame (at 720p) | Active motion                         |
| `Tfing` fingertip motion |                  0.8 px/frame | Scroll/tap micro-motion               |
| `Tmot_low`               |                  0.7 px/frame | On-lap weaker motion                  |
| `Tfing_low`              |                  0.5 px/frame | On-lap weaker finger motion           |
| `Tmot_static`            | 0.4 px/frame for ≥ `S_static` | Static filter                         |
| `S_static`               |           20 frames (≈0.67 s) | Static timeout                        |
| `M_on` / `M_off`         |                 6 / 12 frames | Hysteresis (turn on fast, off slower) |
| Occlusion keep-alive     |                         0.5 s | Short occlusions tolerated            |
| Track max\_age           |                     15 frames | ByteTrack/DeepSORT param              |

> **Note:** If FPS ≠ 30, scale frame counts by (FPS/30). If processing at lower than native FPS, scale thresholds accordingly.

---

## 5) False Positive Defenses (thinking notes)

* **Static veto:** low motion + no hand proximity ⇒ ignore (even if perfectly shaped phone).
* **Hard negatives:** mounts, glossy surfaces, remote controls, wallets, power banks → use to tune thresholds & train ML stage.
* **Near-face strictness:** require *tight* normalized distance + dwell; otherwise many mounted devices near mirrors get flagged.
* **Aspect & size priors:** combat book/tablet confusions; combine with hand proximity to override when true positives.
* **Temporal voting:** require consistency over window W; avoid single-frame spikes.

---

## 6) Performance & Scheduling Strategy

* **Model choices (MVP):**

  * Phones: **YOLOv8n/s** (export to **ONNX**/TRT; FP16 if GPU).
  * Hands: **MMPose** (Lite-HRNet/MobileNetV2); reduce keypoints to wrist + thumb/index if available.
  * Face: **RetinaFace 0.25**.
* **Scheduling:** detect every **K** frames; track each frame; adaptive **K** (raise K if FPS dips).
* **Inference accel:** **ONNX Runtime** (CPU/GPU), **TensorRT** (NVIDIA). Quantize INT8 if calibration data available.
* **Memory:** pre-allocate tensors; avoid image copies; use pinned memory when possible.
* **ROI focus:** when a face is present, prioritize phone/hand search in torso/face ROI for speed.

---

## 7) Logging & Summary (optional enhancement)

**Event schema (per episode):**

```json
{
  "start_time": 12.40,
  "end_time": 18.70,
  "duration": 6.30,
  "mode": "in_hand|near_face|on_lap",
  "mean_confidence": 0.92
}
```

**Frame log (optional):** phone bbox, usage\_prob, hand proximity, motion stats.

**Summary report:**

* Total usage time; % of video; count of episodes; per-mode breakdown.
* Timeline sparkline of `usage_prob` across video.
* **Tech:** **Pandas** for aggregation; **Matplotlib/Plotly** for visuals; export JSON/CSV.

---

## 8) Evaluation Plan (how we’ll know it works)

**Episode-level metrics** (time IoU ≥ 0.5 for true positive episode): precision/recall/F1.
**Box quality**: mean IoU for phone boxes **when ON**.
**Flicker score**: state toggles per minute (lower is better); penalize chattering.
**Runtime**: processed FPS ≥ input FPS on target hardware.
**A/B**: rules-only vs rules+temporal-classifier; track false positive rate vs missed episodes.

**Acceptance targets (initial):**

* Episode F1 ≥ 0.85 on validation.
* FP episodes ≤ 1 per 30-min video.
* Flicker ≤ 2 transitions/minute (median).

---

## 9) Data Strategy & Annotation

* **Labels:** usage episodes (start/end), phone/hand/face boxes on sparse keyframes; interpolate where sensible.
* **Diversity:** lighting, camera positions, subjects, phone colors/cases, occluders (bags, sleeves), reflections.
* **Hard negative mining:** explicitly collect mounts, desk phones, glossy items, remotes, wallets.
* **Augmentations:** motion blur, noise, partial occlusions, low light, compression artifacts.
* **Validation:** keep a holdout from each domain (vehicle cabin, office, home) to test generalization.

---

## 10) Edge Cases & Pre-Decisions

* Mounted phone near face but **no hand**: require very tight near-face + dwell; otherwise **ignore**.
* Speakerphone on desk: **ignore** (no hand + static).
* Earbuds (no phone visible): **out of scope**.
* Two phones: track both independently; no identity needed.
* Gloves: rely more on phone motion & gross hand box than fingertips.

---

## 11) Tech Stack (condensed)

| Function            | Preferred Tech                      | Alternatives            |
| ------------------- | ----------------------------------- | ----------------------- |
| Video I/O & Mux     | **FFmpeg**, OpenCV/PyAV             | MoviePy                 |
| Phone Detection     | **YOLOv8n/s**, YOLO-NAS             | Detectron2, MMDetection |
| Hand Keypoints      | **MMPose (Lite-HRNet/MobileNetV2)** | OpenPose, YOLO-Pose     |
| Face Detection      | **RetinaFace 0.25**                 | YOLO-face, DSFD         |
| Tracking            | **ByteTrack**                       | DeepSORT, NorFair       |
| Motion              | **OpenCV Farneback**                | RAFT, KLT               |
| Temporal Classifier | **XGBoost/LightGBM**                | PyTorch TCN/LSTM        |
| Overlays            | **OpenCV**                          | –                       |
| Audio/Export        | **FFmpeg**                          | –                       |
| Logs/Reports        | **Pandas** + Matplotlib/Plotly      | –                       |

---

## 12) If MediaPipe Were Allowed (suggested usage)

* Replace **MMPose (hands)** with **MediaPipe Hands** for fast CPU keypoints.
* Replace **RetinaFace** with **MediaPipe FaceMesh / Face Detection** for CPU-friendly face boxes/landmarks.
* Pros: cross-platform, low-latency CPU; Cons: licensing/packaging + less flexible training.

---

## 13) Phased Delivery (milestones)

**Phase 1 — MVP (2–4 days):**

* YOLO (phones) + MMPose (hands) + RetinaFace (face) + ByteTrack.
* Rule-based logic + hysteresis; overlays; audio preserved.
* Basic logs (episode start/end).

**Phase 2 — Robustness (1–2 weeks):**

* Tune thresholds per camera domain; hard-negative mining; improved lap heuristic.
* Occlusion keep-alive; adaptive K; performance profiling and INT8/FP16.
* Add summary report; timeline visualization.

**Phase 3 — ML Classifier (1–2 weeks):**

* Feature extraction at scale; train XGBoost/Lightweight TCN.
* Integrate with guardrails; calibrate decision thresholds for precision-first.

---

## 14) Implementation Notes & Gotchas

* Normalize distances and speeds to make thresholds **resolution invariant**.
* Maintain a single **timebase** (pts) so logs align with frames after any dropped/duplicated frames.
* Be strict with **hysteresis** to avoid flicker; err on the side of delayed ON vs premature ON.
* Prefer **centroid velocity** first; add optical flow only if necessary for accuracy.
* Keep original **audio** untouched; only mux at the end to avoid drift.
* Instrument early with simple counters: FPS, detector latency, tracker latency, % frames detected vs tracked.
* Build a **small test clip suite** that you rerun after each change.

---

## 15) Quick Setup Checklist (MVP)

* [ ] Export YOLO phone model to **ONNX**; validate with **ONNX Runtime**.
* [ ] Pick **MMPose** lightweight hand model; verify keypoints on sample frames.
* [ ] Integrate **RetinaFace** for face boxes.
* [ ] Wire **ByteTrack** over phone/hand/face detections.
* [ ] Implement feature buffers (W=0.5 s) and rule logic with hysteresis.
* [ ] Overlay only when **active\_usage=ON**; label with usage probability.
* [ ] Preserve audio with **FFmpeg**; verify identical FPS/resolution.
* [ ] Log episodes; generate a CSV/JSON and a 1-pager summary.

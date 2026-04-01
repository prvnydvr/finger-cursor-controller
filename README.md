# 🖐️ Gesture Based Human Computer Interaction System

> Control your entire computer with just your hand — no mouse, no touchpad, just gestures and voice.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ What It Does

Uses your **webcam + AI hand tracking** to turn gestures and voice into full mouse/keyboard control. No special hardware needed.

---

## 🤌 Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Move cursor |
| 🤏 Pinch (quick) | Left click |
| 🤏 Pinch (hold 0.4s) | Drag |
| ✌️ Two fingers | Right click |
| ✊ Fist | Scroll |
| 🤙 Pinky only | Double click |
| 🖐️ Open hand | Freeze / unfreeze cursor |

---

## 🎙️ Voice Commands

| Say | Action |
|---|---|
| `"click"` | Left click |
| `"right click"` | Right click |
| `"double click"` | Double click |
| `"scroll up"` | Scroll up |
| `"scroll down"` | Scroll down |
| `"copy"` | Ctrl/Cmd + C |
| `"paste"` | Ctrl/Cmd + V |
| `"undo"` | Ctrl/Cmd + Z |
| `"screenshot"` | Take screenshot |
| `"freeze"` | Toggle cursor freeze |

---

## ⚙️ How It Works

- **Hand Tracking** — Google MediaPipe Hand Landmarker (21 keypoints per hand)
- **Smoothing** — One Euro Filter eliminates hand tremor without adding lag
- **Ghost Click Prevention** — Custom `ClickAccuracyEngine` requires gesture confidence ≥ 85% before firing
- **Voice** — Google Speech Recognition via `SpeechRecognition` library
- **Async pipeline** — Camera, ML inference, and voice run on separate threads for low latency

---

## 🚀 Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/finger-cursor-controller.git
cd finger-cursor-controller
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe pyautogui SpeechRecognition pyaudio
```

> **macOS users:** If `pyaudio` fails, run `brew install portaudio` first.

> **Windows users:** If `pyaudio` fails, install from [this wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### 3. Run
```bash
python finger_cursor.py
```

> The hand landmark model (~30 MB) downloads automatically on first run.

---

## 🖥️ Requirements

- Python 3.8+
- Webcam
- Microphone (for voice commands)
- macOS or Windows

---

## 📁 Project Structure

```
finger-cursor-controller/
├── finger_cursor.py      # Main script
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🔧 Tuning

You can tweak these constants at the top of `finger_cursor.py`:

| Constant | Default | Description |
|---|---|---|
| `PINCH_THRESHOLD` | `0.040` | How close fingers must be to detect pinch |
| `DRAG_HOLD_TIME` | `0.40s` | How long to hold pinch before drag starts |
| `CLICK_COOLDOWN` | `0.35s` | Min time between clicks |
| `SCROLL_SENSITIVITY` | `18` | Scroll speed multiplier |

---

## ⚠️ Known Limitations

- Requires good lighting for accurate tracking
- Voice commands need internet (Google Speech API)
- One hand tracked at a time

---

## 🙋 Author

Made by a 12th grader who thought it'd be cool to throw away the mouse.  
Feel free to open issues or PRs!

---

## 📄 License

MIT — use it, fork it, build on it.

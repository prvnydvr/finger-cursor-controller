"""
Finger Cursor Controller
========================
ok so basically this lets you control your mouse with just your hand
index finger = move, pinch = click, fist = scroll, etc
voice commands also work if ur mic isnt trash

gestures:
  index only   -> move cursor
  pinch quick  -> left click
  pinch hold   -> drag (hold 0.4s)
  two fingers  -> right click
  fist         -> scroll (move hand up/down)
  pinky only   -> double click
  open hand    -> freeze cursor (toggle)

voice stuff:
  "click", "right click", "double click"
  "scroll up", "scroll down"
  "copy", "paste", "undo"
  "screenshot", "freeze"

to run:
  pip install opencv-python mediapipe pyautogui SpeechRecognition pyaudio
  python finger_cursor.py

press Q to quit
"""

import cv2
import pyautogui
import numpy as np
import time
import platform
import collections
import threading
import queue
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# mac or windows
IS_MAC     = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
CMD_KEY    = "command" if IS_MAC else "ctrl"

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# tweak these if things feel off
MARGIN             = 0.15    # how much edge to ignore (fraction)
PINCH_THRESHOLD    = 0.040   # how close fingers gotta be for pinch
DRAG_HOLD_TIME     = 0.40    # seconds to hold pinch before drag kicks in
CLICK_COOLDOWN     = 0.35    # so it doesnt spam clicks
SCROLL_SENSITIVITY = 18      # higher = faster scroll
GESTURE_CONFIRM    = 5       # frames before gesture is "confirmed"
DEAD_ZONE          = 5       # pixels, stops cursor going crazy on small tremors
MIN_CONFIDENCE     = 0.80    # ignore bad detections below this

# one euro filter settings — basically smooths the cursor movement
OEF_MINCUTOFF = 1.2
OEF_BETA      = 0.008
OEF_DCUTOFF   = 1.0


# smooths out cursor movement so it doesnt jitter like crazy
class OneEuroFilter:
    def __init__(self, mincutoff=OEF_MINCUTOFF, beta=OEF_BETA, dcutoff=OEF_DCUTOFF):
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x        = None
        self._dx       = 0.0
        self._t        = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-9))

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._t = x, t
            return x
        dt   = t - self._t
        dx   = (x - self._x) / max(dt, 1e-9)
        a_d  = self._alpha(self.dcutoff, dt)
        dx_h = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.mincutoff + self.beta * abs(dx_h)
        a    = self._alpha(cutoff, dt)
        x_h  = a * x + (1 - a) * self._x
        self._x, self._dx, self._t = x_h, dx_h, t
        return x_h

filter_x = OneEuroFilter()
filter_y = OneEuroFilter()


# this thing makes sure clicks only fire when ur actually doing the gesture
# not when ur hand just passes through it for a frame
# saved me from so many ghost clicks fr
class ClickAccuracyEngine:
    def __init__(self, threshold=0.85, decay=0.18, gain=0.30):
        self.threshold = threshold
        self.decay     = decay
        self.gain      = gain
        self._scores   = collections.defaultdict(float)
        self._fired    = set()

    def update(self, active: set):
        for g in set(self._scores.keys()) | active:
            if g in active:
                self._scores[g] = min(1.0, self._scores[g] + self.gain)
            else:
                self._scores[g] = max(0.0, self._scores[g] - self.decay)
                if self._scores[g] == 0.0:
                    self._fired.discard(g)

    def ready(self, gesture: str) -> bool:
        if self._scores[gesture] >= self.threshold and gesture not in self._fired:
            self._fired.add(gesture)
            return True
        return False

    def reset(self, gesture: str):
        self._scores[gesture] = 0.0
        self._fired.discard(gesture)

    def score(self, gesture: str) -> float:
        return self._scores.get(gesture, 0.0)

accuracy_engine = ClickAccuracyEngine()

# global state stuff
cursor_x, cursor_y     = float(SCREEN_W // 2), float(SCREEN_H // 2)
dragging               = False
pinch_start_time       = None
scroll_ref_y           = None
cursor_frozen          = False

last_click_time        = 0.0
last_right_click_time  = 0.0
last_double_click_time = 0.0

gesture_history = collections.deque(maxlen=12)
gesture_buffer  = collections.deque(maxlen=GESTURE_CONFIRM)
voice_queue: queue.Queue = queue.Queue()

# mediapipe landmark indices, just naming them so i dont lose my mind
BaseOptions           = mp_python.BaseOptions
HandLandmarker        = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode     = vision.RunningMode

WRIST=0; THUMB_IP=3; THUMB_TIP=4
INDEX_PIP=6;  INDEX_TIP=8
MIDDLE_PIP=10; MIDDLE_TIP=12
RING_PIP=14;  RING_TIP=16
PINKY_PIP=18; PINKY_TIP=20


# distance between two points, classic
def dist(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

# get landmark pixel coords
def lm(lms, idx, w, h):
    p = lms[idx]
    return int(p.x * w), int(p.y * h)

# maps camera coords to screen coords
# also flips x because camera is mirrored
def map_screen(x, y, fw, fh):
    x = fw - x
    lo, hi = MARGIN, 1.0 - MARGIN
    nx = np.clip((x/fw - lo) / (hi - lo), 0.0, 1.0)
    ny = np.clip((y/fh - lo) / (hi - lo), 0.0, 1.0)
    return nx * SCREEN_W, ny * SCREEN_H

# stops cursor twitching when hand is basically still
def dead_zone(nx, ny, px, py):
    dx, dy = nx - px, ny - py
    mag = np.hypot(dx, dy)
    if mag < DEAD_ZONE:
        return px, py
    ease = (mag - DEAD_ZONE) / mag
    return px + dx * ease, py + dy * ease


# figures out which fingers are up
def fingers_up(lms_list, w, h):
    tt = lm(lms_list, THUMB_TIP, w, h)
    ti = lm(lms_list, THUMB_IP,  w, h)
    wr = lm(lms_list, WRIST,     w, h)
    thumb_up = dist(tt, wr) > dist(ti, wr)
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    rest = [lm(lms_list, t, w, h)[1] < lm(lms_list, p, w, h)[1]
            for t, p in zip(tips, pips)]
    return [thumb_up] + rest  # [thumb, index, middle, ring, pinky]

# maps finger state to a gesture name
def detect_gesture(lms_list, w, h):
    up = fingers_up(lms_list, w, h)
    it = lm(lms_list, INDEX_TIP,  w, h)
    tt = lm(lms_list, THUMB_TIP,  w, h)
    pd = dist(it, tt) / np.hypot(w, h)

    if not any(up[1:]):                                         return 'fist'
    if up[1] and pd < PINCH_THRESHOLD:                          return 'pinch'
    if up[4] and not up[1] and not up[2] and not up[3]:         return 'pinky'
    if up[1] and up[2] and not up[3] and not up[4]:             return 'two_fingers'
    if up[1] and up[2] and up[3] and up[4]:                     return 'open'
    if up[1] and not up[2] and not up[3] and not up[4]:         return 'point'
    return 'idle'

# needs GESTURE_CONFIRM frames of same gesture before locking it in
# stops random flickers from doing stuff
def confirmed_gesture(raw):
    gesture_buffer.append(raw)
    if len(gesture_buffer) == GESTURE_CONFIRM and len(set(gesture_buffer)) == 1:
        return gesture_buffer[-1]
    return gesture_buffer[-1] if gesture_buffer else 'idle'


# runs in background thread, listens for voice commands
def voice_listener(q: queue.Queue):
    try:
        import speech_recognition as sr
    except ImportError:
        print("voice wont work, install SpeechRecognition and pyaudio")
        return

    COMMANDS = {
        "click":        "click",
        "left click":   "click",
        "right click":  "right_click",
        "double click": "double_click",
        "double tap":   "double_click",
        "scroll up":    "scroll_up",
        "scroll down":  "scroll_down",
        "copy":         "copy",
        "paste":        "paste",
        "undo":         "undo",
        "screenshot":   "screenshot",
        "freeze":       "freeze",
        "stop":         "freeze",
        "resume":       "freeze",
    }

    recogniser = sr.Recognizer()
    recogniser.energy_threshold         = 300
    recogniser.dynamic_energy_threshold = True
    recogniser.pause_threshold          = 0.5

    try:
        with sr.Microphone() as source:
            recogniser.adjust_for_ambient_noise(source, duration=1)
            print("mic ready, voice commands active")
            while True:
                try:
                    audio = recogniser.listen(source, timeout=5, phrase_time_limit=3)
                    text  = recogniser.recognize_google(audio).lower().strip()
                    for phrase, cmd in COMMANDS.items():
                        if phrase in text:
                            q.put(cmd)
                            break
                except Exception:
                    pass
    except Exception as e:
        print(f"mic error: {e}")

def execute_voice(cmd: str):
    global cursor_frozen
    actions = {
        "click":        lambda: pyautogui.click(),
        "right_click":  lambda: pyautogui.rightClick(),
        "double_click": lambda: pyautogui.doubleClick(),
        "scroll_up":    lambda: pyautogui.scroll(10),
        "scroll_down":  lambda: pyautogui.scroll(-10),
        "copy":         lambda: pyautogui.hotkey(CMD_KEY, "c"),
        "paste":        lambda: pyautogui.hotkey(CMD_KEY, "v"),
        "undo":         lambda: pyautogui.hotkey(CMD_KEY, "z"),
        "screenshot":   lambda: (pyautogui.hotkey("command","shift","3")
                                 if IS_MAC else
                                 pyautogui.hotkey("win","shift","s")),
    }
    if cmd == "freeze":
        cursor_frozen = not cursor_frozen
    elif cmd in actions:
        actions[cmd]()
    return cmd.replace("_", " ")


# colors for each gesture in BGR
COLORS = {
    'point':       (0,   255, 100),
    'pinch':       (0,   120, 255),
    'two_fingers': (255, 100,   0),
    'fist':        (255,  50,  50),
    'pinky':       (200,   0, 255),
    'open':        (200, 200,  50),
    'idle':        (160, 160, 160),
}

# short labels shown next to gesture trail
GESTURE_ICONS = {
    'point':       '[I]',
    'pinch':       '[P]',
    'two_fingers': '[2]',
    'fist':        '[F]',
    'pinky':       '[K]',
    'open':        '[O]',
    'idle':        '[ ]',
}

def draw_landmarks(frame, lms_list, w, h):
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
             (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    pts = [lm(lms_list, i, w, h) for i in range(21)]
    for a, b in conns:
        cv2.line(frame, pts[a], pts[b], (80, 180, 80), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (60, 160, 60), 1)

def draw_confidence_bar(frame, gesture, x, y):
    score  = accuracy_engine.score(gesture)
    bar_w  = 130
    filled = int(score * bar_w)
    cv2.rectangle(frame, (x, y), (x+bar_w, y+8), (40, 40, 40), -1)
    color = (0, 220, 100) if score >= 0.85 else (0, 150, 255)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x+filled, y+8), color, -1)
    cv2.rectangle(frame, (x, y), (x+bar_w, y+8), (100, 100, 100), 1)

def draw_ui(frame, gesture, sx, sy, dragging, frozen, last_voice, fps=0):
    fh, fw = frame.shape[:2]
    color  = COLORS.get(gesture, (160, 160, 160))
    icon   = GESTURE_ICONS.get(gesture, '[ ]')

    # top left status box
    cv2.rectangle(frame, (0, 0), (270, 125), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, 0), (270, 125), (55, 55, 55), 1)

    label = f"{icon} {gesture.upper().replace('_', ' ')}"
    if dragging: label += "  [DRAG]"
    if frozen:   label += "  [FROZEN]"

    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(frame, f"Cursor  {int(sx):4d} x {int(sy):4d}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
    cv2.putText(frame, "Confidence", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)
    draw_confidence_bar(frame, gesture, 10, 78)

    if last_voice:
        cv2.putText(frame, f"Voice: {last_voice}", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    # fps counter top right
    cv2.putText(frame, f"FPS {fps}", (fw - 70, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # gesture trail on left side
    for i, g in enumerate(reversed(gesture_history)):
        alpha = max(0.15, 1.0 - i * 0.10)
        c = tuple(int(v * alpha) for v in COLORS.get(g, (160, 160, 160)))
        tag = GESTURE_ICONS.get(g, '[ ]')
        cv2.putText(frame, f"{tag} {g}", (10, 145 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1)

    # right side quick reference panel
    guide = [
        ("[I] Index",   "Move"),
        ("[P] Pinch",   "Click"),
        ("[P] Hold",    "Drag"),
        ("[2] Two",     "R-Click"),
        ("[F] Fist",    "Scroll"),
        ("[K] Pinky",   "DblClick"),
        ("[O] Open",    "Freeze"),
    ]
    gx = fw - 155
    cv2.rectangle(frame, (gx - 8, 0), (fw, len(guide) * 22 + 14), (18, 18, 18), -1)
    cv2.rectangle(frame, (gx - 8, 0), (fw, len(guide) * 22 + 14), (55, 55, 55), 1)
    for i, (g, a) in enumerate(guide):
        cy = 18 + i * 22
        cv2.putText(frame, g, (gx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.putText(frame, a, (gx + 85, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color if gesture in g.lower() else (100, 100, 100), 1)

    # bottom instructions bar
    lines = [
        "Q = Quit    Open Hand = Freeze    Pinky = Double Click",
        "Fist = Scroll    Two Fingers = Right Click    Pinch Hold = Drag",
    ]
    panel_y = fh - len(lines) * 18 - 10
    cv2.rectangle(frame, (0, panel_y - 6), (fw, fh), (18, 18, 18), -1)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (8, panel_y + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 110), 1)

    # active zone box so you know where to move ur hand
    mx1 = int(MARGIN * fw); my1 = int(MARGIN * fh)
    mx2 = int((1 - MARGIN) * fw); my2 = int((1 - MARGIN) * fh)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (70, 70, 70), 1)
    cv2.putText(frame, "active zone", (mx1 + 4, my1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

    # blue tint + text when frozen
    if frozen:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "CURSOR FROZEN", (fw // 2 - 110, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)


def main():
    global cursor_x, cursor_y, dragging, pinch_start_time
    global scroll_ref_y, cursor_frozen
    global last_click_time, last_right_click_time, last_double_click_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("cant open webcam")
        if IS_MAC:
            print("go to System Settings > Privacy > Camera and allow Terminal")
        return

    # download model if not cached
    import urllib.request, os
    model_path = os.path.expanduser("~/.cache/mediapipe_hand_landmarker.task")
    if not os.path.exists(model_path):
        print("downloading hand model, ~30mb, only happens once...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("model ready")

    latest_result = [None]
    def on_result(result, _img, _ts):
        latest_result[0] = result

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_hands=1,
        min_hand_detection_confidence=MIN_CONFIDENCE,
        min_hand_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=0.65,
    )

    # voice runs separately so it doesnt block main loop
    threading.Thread(target=voice_listener, args=(voice_queue,), daemon=True).start()

    frame_ts            = 0
    last_voice          = ""
    voice_display_until = 0.0

    # for fps calc
    fps_times  = collections.deque(maxlen=30)
    prev_time  = time.time()
    fps        = 0

    print("running! point your index finger to move cursor")
    print("press Q to quit\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw = frame.shape[:2]
            now    = time.time()

            # fps
            fps_times.append(now - prev_time)
            prev_time = now
            if fps_times:
                fps = int(1.0 / (sum(fps_times) / len(fps_times)))

            # send frame to mediapipe
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 1
            landmarker.detect_async(mp_image, frame_ts)

            result  = latest_result[0]
            gesture = 'idle'

            if result and result.hand_landmarks:
                lms_list = result.hand_landmarks[0]
                draw_landmarks(frame, lms_list, fw, fh)

                raw     = detect_gesture(lms_list, fw, fh)
                gesture = confirmed_gesture(raw)
                gesture_history.append(gesture)
                accuracy_engine.update({gesture})

                it = lm(lms_list, INDEX_TIP, fw, fh)
                wy = lm(lms_list, WRIST,     fw, fh)[1]

                # dot on index tip
                cv2.circle(frame, (fw - it[0], it[1]), 10,
                           COLORS.get(gesture, (0, 255, 0)), -1)

                if not cursor_frozen:

                    if gesture == 'point':
                        rx, ry = map_screen(it[0], it[1], fw, fh)
                        sx = filter_x(rx, now)
                        sy = filter_y(ry, now)
                        cursor_x, cursor_y = dead_zone(sx, sy, cursor_x, cursor_y)
                        pyautogui.moveTo(int(cursor_x), int(cursor_y))
                        scroll_ref_y = None

                    elif gesture == 'pinch':
                        rx, ry = map_screen(it[0], it[1], fw, fh)
                        sx = filter_x(rx, now)
                        sy = filter_y(ry, now)
                        cursor_x, cursor_y = dead_zone(sx, sy, cursor_x, cursor_y)
                        pyautogui.moveTo(int(cursor_x), int(cursor_y))
                        if pinch_start_time is None:
                            pinch_start_time = now
                        held = now - pinch_start_time
                        if held >= DRAG_HOLD_TIME and not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        scroll_ref_y = None

                    elif gesture == 'two_fingers':
                        rx, ry = map_screen(it[0], it[1], fw, fh)
                        cursor_x = filter_x(rx, now)
                        cursor_y = filter_y(ry, now)
                        pyautogui.moveTo(int(cursor_x), int(cursor_y))
                        if accuracy_engine.ready('two_fingers') and \
                           (now - last_right_click_time) > CLICK_COOLDOWN:
                            pyautogui.rightClick()
                            last_right_click_time = now
                        scroll_ref_y = None

                    elif gesture == 'pinky':
                        if accuracy_engine.ready('pinky') and \
                           (now - last_double_click_time) > CLICK_COOLDOWN:
                            pyautogui.doubleClick()
                            last_double_click_time = now
                        scroll_ref_y = None

                    elif gesture == 'fist':
                        if scroll_ref_y is None:
                            scroll_ref_y = wy
                        else:
                            delta = scroll_ref_y - wy
                            amt   = int(delta / fh * SCROLL_SENSITIVITY * 10)
                            if abs(amt) > 0:
                                pyautogui.scroll(amt)
                            scroll_ref_y = wy

                    elif gesture == 'open':
                        if accuracy_engine.ready('open'):
                            cursor_frozen = True
                        scroll_ref_y = None

                # release drag if pinch ended
                if gesture != 'pinch':
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    if pinch_start_time is not None:
                        held = now - pinch_start_time
                        if held < DRAG_HOLD_TIME and \
                           (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                    pinch_start_time = None

                # reset confidence when gesture changes
                for g in ['two_fingers', 'pinky', 'open']:
                    if gesture != g:
                        accuracy_engine.reset(g)

                # any non-open gesture unfreezes the cursor
                if cursor_frozen and gesture not in ('open', 'idle'):
                    cursor_frozen = False

            else:
                # no hand found, clean up any active states
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                pinch_start_time = None
                gesture_buffer.clear()
                scroll_ref_y = None

            # handle voice commands from queue
            while not voice_queue.empty():
                cmd = voice_queue.get_nowait()
                last_voice = execute_voice(cmd)
                voice_display_until = now + 2.5
            if now > voice_display_until:
                last_voice = ""

            # flip and draw everything
            frame = cv2.flip(frame, 1)
            draw_ui(frame, gesture, cursor_x, cursor_y, dragging, cursor_frozen, last_voice, fps)
            cv2.imshow("Finger Cursor Controller  —  Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("stopped.")


if __name__ == "__main__":
    main()
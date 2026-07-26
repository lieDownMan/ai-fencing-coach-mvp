"""Mac-camera tuning server — remote threshold tuning for the iPhone app.

The Mac films you with its webcam; your PHONE's browser is the remote
control (live view + big metric readout + threshold slider + voice cue).
No walking to the camera needed.

The window metrics implemented here MIRROR the Flutter engine's
``computeWindowMetrics`` (frontend/lib/heuristics/heuristics_engine.dart):
same metric names, same math, same parameter names, same defaults — so a
number tuned here pastes 1:1 into ``HeuristicsConfig``.

Run (from repo root; first run asks for macOS camera permission):
    venv/bin/python backend/tuning_server.py            # webcam 0
    venv/bin/python backend/tuning_server.py --source path/to/clip.mp4
    venv/bin/python backend/tuning_server.py --self-test

Then open  http://<Mac-IP>:8123  on the phone (same Wi-Fi).
Tuned values persist to backend/tuning_overrides.json; the page's
「複製 Dart 參數」 button gives the snippet to paste into HeuristicsConfig.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

OVERRIDES_PATH = BACKEND_DIR / "tuning_overrides.json"

# ---------------------------------------------------------------------------
# Config defaults — MUST match HeuristicsConfig in heuristics_engine.dart
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "bounceRatioThreshold": 0.13,
    "lungeKneeMinAngleDeg": 153.5,
    "guardDroppedSeconds": 3.0,
    "guardDroppedFreeBoutingSeconds": 3.0,
    "footBeforeHandMinDisplacement": 0.03,
    "footBeforeHandLeadSeconds": 0.10,
    "stanceTooHighAngleDeg": 170.0,
    "incompleteArmExtensionAngleDeg": 101.0,
    "overParryTorsoRatioThreshold": 0.54,
    "stepShoulderProxyMultiplier": 2.5,
    "wideStepRatioThreshold": 1.72,
    "narrowStepRatioThreshold": 0.9,
    "stepSustainedSeconds": 0.30,
    "comForwardLeanDeg": 14.0,
    "comBackwardLeanDeg": 5.5,
    "comSustainedSeconds": 0.30,
}

# Mirrors kTuningSpecs in frontend/lib/tuning/tuning_specs.dart.
# direction: "above" = metric > threshold triggers; "below" = metric < threshold.
SPECS = [
    {"error_key": "stance_too_high", "label": "預備姿勢沒蹲好 Stance Too High",
     "metric": "avg_front_knee_angle_deg", "param": "stanceTooHighAngleDeg",
     "direction": "above", "min": 140, "max": 180, "unit": "°", "decimals": 1,
     "hint": "維持 en garde 看膝角。180=站直、標準蹲姿 120–140。"},
    {"error_key": "lunge_overextension", "label": "長刺過度前傾 Lunge Overextension",
     "metric": "lunge_knee_angle_deg", "param": "lungeKneeMinAngleDeg",
     "direction": "below", "min": 130, "max": 170, "unit": "°", "decimals": 1,
     "hint": "做弓步，數字是最深那幀的前膝角。90=小腿垂直。"},
    {"error_key": "incomplete_arm_extension", "label": "手沒有伸直 Incomplete Extension",
     "metric": "arm_extension_angle_deg", "param": "incompleteArmExtensionAngleDeg",
     "direction": "below", "min": 70, "max": 120, "unit": "°", "decimals": 1,
     "hint": "出手刺擊，數字是最遠那幀的手臂角。伸直=170–180。"},
    {"error_key": "bounce_excessive", "label": "步伐上下浮動 Excessive Bounce",
     "metric": "bounce_ratio", "param": "bounceRatioThreshold",
     "direction": "above", "min": 0.05, "max": 0.60, "unit": "", "decimals": 2,
     "hint": "做步伐，數字是骨盆起伏佔身高比例。"},
    {"error_key": "guard_dropped", "label": "持劍手掉落 Guard Dropped",
     "metric": "guard_below_pelvis_max_run_s", "param": "guardDroppedSeconds",
     "direction": "above", "min": 0.50, "max": 5.00, "unit": "s", "decimals": 2,
     "hint": "手垂低於骨盆，數字是連續低垂秒數。"},
    {"error_key": "over_parrying", "label": "防守動作太大 Over-Parrying",
     "metric": "parry_sweep_torso_ratio", "param": "overParryTorsoRatioThreshold",
     "direction": "above", "min": 0.20, "max": 2.50, "unit": "×軀幹", "decimals": 2,
     "hint": "做防守揮劍，數字是手腕橫掃範圍 ÷ 軀幹長。"},
    {"error_key": "narrow_step", "label": "步伐太小 Narrow Step",
     "metric": "step_ratio_median", "param": "narrowStepRatioThreshold",
     "direction": "below", "min": 0.30, "max": 2.00, "unit": "×肩寬", "decimals": 2,
     "hint": "維持窄站姿看數字。正常 en garde 約 1.5–2.5。"},
    {"error_key": "wide_step", "label": "步伐太大 Wide Step",
     "metric": "step_ratio_median", "param": "wideStepRatioThreshold",
     "direction": "above", "min": 1.00, "max": 2.00, "unit": "×肩寬", "decimals": 2,
     "hint": "維持寬站姿看數字。正常 en garde 約 1.5–2.5。"},
    {"error_key": "center_of_mass_in_front", "label": "重心向前 CoM Forward",
     "metric": "torso_lean_deg_median", "param": "comForwardLeanDeg",
     "direction": "above", "min": 5, "max": 45, "unit": "°", "decimals": 1,
     "hint": "軀幹前傾角（骨盆→肩膀 vs 鉛直線）。0=直立、正=朝對手傾。"},
    {"error_key": "center_of_mass_leaning_backward", "label": "重心向後 CoM Backward",
     "metric": "torso_lean_deg_median", "param": "comBackwardLeanDeg",
     "direction": "below", "min": -30, "max": 15, "unit": "°", "decimals": 1,
     "hint": "軀幹傾角，負=向後仰。低於閾值觸發。"},
    {"error_key": "foot_before_hand", "label": "手腳順序錯誤 Foot Before Hand",
     "metric": "foot_hand_lead_s", "param": "footBeforeHandLeadSeconds",
     "direction": "above", "min": 0.00, "max": 0.50, "unit": "s", "decimals": 2,
     "hint": "做刺靶動作才有值：正數=腳比手先動幾秒。"},
]

SPEC_BY_KEY = {s["error_key"]: s for s in SPECS}

BOUNCE_MIN_PELVIS_SAMPLES = 5
OVER_PARRY_MIN_WRIST_SAMPLES = 5
STEP_MIN_SHOULDER_WIDTH = 0.01
COM_MIN_BASE_WIDTH = 0.01

# ---------------------------------------------------------------------------
# Geometry (mirror of the Dart engine)
# ---------------------------------------------------------------------------


def calc_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    nba = math.hypot(*ba)
    nbc = math.hypot(*bc)
    if nba < 1e-8 or nbc < 1e-8:
        return 180.0
    cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (nba * nbc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def pelvis_center(skel):
    lh, rh = skel.get("left_hip"), skel.get("right_hip")
    if lh is None or rh is None:
        return None
    return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)


def front_limbs(target_side):
    side = "right" if target_side == "left" else "left"
    return {
        "hip": f"{side}_hip", "knee": f"{side}_knee", "ankle": f"{side}_ankle",
        "wrist": "front_wrist", "elbow": "front_elbow",
        "shoulder": "front_shoulder",
    }


def back_ankle_name(target_side):
    return "left_ankle" if target_side == "left" else "right_ankle"


def compute_window_metrics(skeletons, fps, target_side, config):
    """Python mirror of HeuristicsEngine.computeWindowMetrics (Dart)."""
    m = {}
    if not skeletons:
        return m
    if not math.isfinite(fps) or fps <= 1.0:
        fps = 30.0
    limbs = front_limbs(target_side)
    back_key = back_ankle_name(target_side)

    pelvis_ys, all_ys = [], []
    for skel in skeletons:
        pc = pelvis_center(skel)
        if pc is not None:
            pelvis_ys.append(pc[1])
        all_ys.extend(v[1] for v in skel.values())
    if len(pelvis_ys) >= BOUNCE_MIN_PELVIS_SAMPLES and len(all_ys) >= 2:
        bbox = max(all_ys) - min(all_ys)
        if bbox > 1e-4:
            m["bounce_ratio"] = (max(pelvis_ys) - min(pelvis_ys)) / bbox

    knee_angles = []
    for skel in skeletons:
        hip, knee, ankle = (skel.get(limbs["hip"]), skel.get(limbs["knee"]),
                            skel.get(limbs["ankle"]))
        if hip and knee and ankle:
            knee_angles.append(calc_angle(hip, knee, ankle))
    if len(knee_angles) >= 3:
        m["avg_front_knee_angle_deg"] = sum(knee_angles) / len(knee_angles)
        m["min_front_knee_angle_deg"] = min(knee_angles)

    ref_ankle = skeletons[0].get(limbs["ankle"])
    if ref_ankle:
        max_disp, peak = 0.0, skeletons[0]
        for skel in skeletons:
            ankle = skel.get(limbs["ankle"])
            if ankle:
                d = math.hypot(ankle[0] - ref_ankle[0], ankle[1] - ref_ankle[1])
                if d > max_disp:
                    max_disp, peak = d, skel
        hip, knee, ankle = (peak.get(limbs["hip"]), peak.get(limbs["knee"]),
                            peak.get(limbs["ankle"]))
        if hip and knee and ankle:
            m["lunge_knee_angle_deg"] = calc_angle(hip, knee, ankle)

    ref_wrist = skeletons[0].get(limbs["wrist"])
    if ref_wrist:
        max_disp, peak = 0.0, skeletons[0]
        for skel in skeletons:
            wrist = skel.get(limbs["wrist"])
            if wrist:
                d = abs(wrist[0] - ref_wrist[0])
                if d > max_disp:
                    max_disp, peak = d, skel
        sh, el, wr = (peak.get(limbs["shoulder"]), peak.get(limbs["elbow"]),
                      peak.get(limbs["wrist"]))
        if sh and el and wr:
            m["arm_extension_angle_deg"] = calc_angle(sh, el, wr)

    rel_xs, torso_lens = [], []
    for skel in skeletons:
        wrist, pelvis = skel.get(limbs["wrist"]), pelvis_center(skel)
        if wrist is None or pelvis is None:
            continue
        rel_xs.append(wrist[0] - pelvis[0])
        shoulder = skel.get(limbs["shoulder"])
        if shoulder:
            torso_lens.append(
                math.hypot(shoulder[0] - pelvis[0], shoulder[1] - pelvis[1]))
    if len(rel_xs) >= OVER_PARRY_MIN_WRIST_SAMPLES and torso_lens:
        torso = statistics.median(torso_lens)
        if torso > 1e-6:
            m["parry_sweep_torso_ratio"] = (max(rel_xs) - min(rel_xs)) / torso

    sign = _window_facing_sign(skeletons, limbs, back_key, target_side)
    step_ratios, leans = [], []
    for skel in skeletons:
        fa, ba_, pelvis = (skel.get(limbs["ankle"]), skel.get(back_key),
                           pelvis_center(skel))
        shoulder = skel.get(limbs["shoulder"])
        if fa is not None and ba_ is not None and pelvis is not None and shoulder:
            sw = abs(shoulder[0] - pelvis[0]) * config["stepShoulderProxyMultiplier"]
            if sw >= STEP_MIN_SHOULDER_WIDTH:
                step_ratios.append(abs(fa[0] - ba_[0]) / sw)
        # torso lean from vertical (pelvis→front shoulder), + = toward opponent
        if pelvis is not None and shoulder:
            vertical = pelvis[1] - shoulder[1]
            if vertical > 1e-6:
                fwd = sign * (shoulder[0] - pelvis[0])
                leans.append(math.degrees(math.atan2(fwd, vertical)))
    if step_ratios:
        m["step_ratio_min"] = min(step_ratios)
        m["step_ratio_max"] = max(step_ratios)
        m["step_ratio_median"] = statistics.median(step_ratios)
    if leans:
        m["torso_lean_deg_min"] = min(leans)
        m["torso_lean_deg_max"] = max(leans)
        m["torso_lean_deg_median"] = statistics.median(leans)

    run = max_run = 0
    for skel in skeletons:
        wrist, pelvis = skel.get(limbs["wrist"]), pelvis_center(skel)
        if wrist and pelvis and wrist[1] > pelvis[1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    m["guard_below_pelvis_max_run_s"] = max_run / fps

    # foot-before-hand: body-relative forward series + rise-onset from the
    # global peak — mirrors _checkFootBeforeHand / computeWindowMetrics (Dart).
    wrist_rel, ankle_rel = [], []
    for skel in skeletons:
        pelvis = pelvis_center(skel)
        wrist = skel.get(limbs["wrist"])
        ankle = skel.get(limbs["ankle"])
        wrist_rel.append(None if pelvis is None or wrist is None
                         else sign * (wrist[0] - pelvis[0]))
        ankle_rel.append(None if pelvis is None or ankle is None
                         else sign * (ankle[0] - pelvis[0]))
    w_on = _rise_onset(wrist_rel, config["footBeforeHandMinDisplacement"])
    a_on = _rise_onset(ankle_rel, config["footBeforeHandMinDisplacement"])
    if w_on is not None and a_on is not None:
        m["foot_hand_lead_s"] = (w_on - a_on) / fps

    return m


def _window_facing_sign(skeletons, limbs, back_key, target_side):
    """Facing from foot placement: front ankle is always toward the opponent.
    Falls back to the targetSide convention when feet are missing."""
    diffs = []
    for skel in skeletons:
        fa, ba = skel.get(limbs["ankle"]), skel.get(back_key)
        if fa is not None and ba is not None:
            diffs.append(fa[0] - ba[0])
    if not diffs:
        return 1.0 if target_side == "left" else -1.0
    med = statistics.median(diffs)
    if abs(med) < 1e-6:
        return 1.0 if target_side == "left" else -1.0
    return 1.0 if med > 0 else -1.0


def _rise_onset(series, min_rise):
    """Onset index of the final rise toward the global peak (Dart _riseOnset)."""
    peak_idx, peak_val = None, -math.inf
    for i, v in enumerate(series):
        if v is not None and v > peak_val:
            peak_val, peak_idx = v, i
    if peak_idx is None:
        return None
    baseline = min(v for v in series[:peak_idx + 1] if v is not None)
    rise = peak_val - baseline
    if rise < min_rise:
        return None
    onset_level = baseline + 0.1 * rise
    onset = peak_idx
    for i in range(peak_idx, -1, -1):
        v = series[i]
        if v is None:
            continue
        if v <= onset_level:
            break
        onset = i
    return onset


# ---------------------------------------------------------------------------
# Camera / pose loop
# ---------------------------------------------------------------------------

SKELETON_EDGES = [
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class TuningState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = dict(DEFAULT_CONFIG)
        if OVERRIDES_PATH.exists():
            try:
                stored = json.loads(OVERRIDES_PATH.read_text())
                # Stale-schema check: pre-retune files carry removed keys and
                # values chosen against the old defaults — discard wholesale.
                if any(k not in DEFAULT_CONFIG for k in stored):
                    OVERRIDES_PATH.unlink()
                else:
                    self.config.update(
                        {k: float(v) for k, v in stored.items()})
            except Exception:
                pass
        self.error_key = "stance_too_high"
        self.target_side = "left"
        self.metrics = {}
        self.fps = 0.0
        self.triggered = False
        self.person_visible = False
        self.latest_jpeg = None

    def save(self):
        OVERRIDES_PATH.write_text(json.dumps(self.config, indent=2))

    def snippet(self):
        return "".join(f"    this.{k} = {v},\n" for k, v in self.config.items())


class CameraLoop(threading.Thread):
    def __init__(self, state, source):
        super().__init__(daemon=True)
        self.state = state
        self.source = source
        self.is_file = not str(source).isdigit()

    def run(self):
        import cv2
        from src.pose_estimation import PoseEstimator

        estimator = PoseEstimator(backend="ultralytics",
                                  target_side=self.state.target_side)
        cap = cv2.VideoCapture(int(self.source) if not self.is_file
                               else str(self.source))
        if not cap.isOpened():
            print(f"!! cannot open camera/video source: {self.source}")
            return
        print(f"camera loop started (source={self.source})")

        buffer = deque(maxlen=60)
        stamps = deque(maxlen=30)
        frame_i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            estimator.target_side = self.state.target_side
            detections = estimator.extract_frame_fencers(frame)
            h, w = frame.shape[:2]
            skel_px = detections[0]["skeleton"] if detections else None

            if skel_px:
                norm = {k: (v[0] / w, v[1] / h) for k, v in skel_px.items()}
                buffer.append(norm)
                stamps.append(time.monotonic())

            fps = 30.0
            if len(stamps) >= 5 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])

            frame_i += 1
            if frame_i % 3 == 0 and buffer:
                with self.state.lock:
                    cfg = dict(self.state.config)
                    key = self.state.error_key
                    side = self.state.target_side
                metrics = compute_window_metrics(list(buffer), fps, side, cfg)
                spec = SPEC_BY_KEY[key]
                value = metrics.get(spec["metric"])
                thr = cfg[spec["param"]]
                trig = (value is not None and
                        (value > thr if spec["direction"] == "above"
                         else value < thr))
                with self.state.lock:
                    self.state.metrics = metrics
                    self.state.fps = fps
                    self.state.triggered = trig
                    self.state.person_visible = skel_px is not None

            # annotate + publish preview
            if skel_px:
                for a, b in SKELETON_EDGES:
                    pa, pb = skel_px.get(a), skel_px.get(b)
                    if pa and pb:
                        cv2.line(frame, (int(pa[0]), int(pa[1])),
                                 (int(pb[0]), int(pb[1])),
                                 (0, 0, 255) if self.state.triggered
                                 else (80, 230, 80), 3)
            if self.state.triggered:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)
            small = cv2.resize(frame, (640, int(640 * h / w)))
            ok2, jpg = cv2.imencode(".jpg", small,
                                    [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok2:
                self.state.latest_jpeg = jpg.tobytes()

            if self.is_file:
                time.sleep(0.03)


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fencing Tuning Remote</title>
<style>
body{background:#0a0a0f;color:#eee;font-family:-apple-system,sans-serif;
     margin:0;padding:14px;max-width:520px;margin:auto}
select,button{font-size:15px;padding:8px;border-radius:8px;background:#141420;
     color:#eee;border:1px solid #333;width:100%%;margin:4px 0}
img{width:100%%;border-radius:10px;border:1px solid #222}
#big{font-size:52px;font-weight:800;text-align:center;font-variant-numeric:tabular-nums}
#state{text-align:center;font-weight:700;font-size:15px;padding:6px;border-radius:8px}
.trig{background:#3a0000;color:#ff5252;border:1px solid #ff5252}
.ok{background:#0e2a12;color:#69f0ae;border:1px solid #69f0ae}
input[type=range]{width:100%%;accent-color:#ff6600;height:34px}
.thr{text-align:center;color:#aaa;font-size:14px}
.hint{color:#888;font-size:12px;margin:4px 0}
.row{display:flex;gap:8px}
</style></head><body>
<h3 style="margin:4px 0">⚔️ Tuning Remote</h3>
<select id="err"></select>
<div class="hint" id="hint"></div>
<div class="row">
<select id="side" style="flex:1">
  <option value="left">左側擊劍手 (left)</option>
  <option value="right">右側擊劍手 (right)</option>
</select>
<button id="voice" style="flex:1">🔊 語音提示: 開</button>
</div>
<img id="cam" src="/stream">
<div id="state" class="ok">--</div>
<div id="big">--</div>
<div class="thr" id="thr"></div>
<input type="range" id="slider" step="0.01">
<div class="row">
<button id="reset">重設此項</button>
<button id="copy">複製 Dart 參數</button>
</div>
<div class="hint">數值即時比對、閾值放手即存檔（backend/tuning_overrides.json）。
複製的片段貼回 HeuristicsConfig 或傳給 AI。</div>
<script>
const SPECS = %SPECS%;
let voiceOn = true, lastTrig = false, spec = null, dragging=false;
const $ = id => document.getElementById(id);
const errSel = $('err');
SPECS.forEach(s => errSel.add(new Option(s.label, s.error_key)));
function setSpec(key){
  spec = SPECS.find(s => s.error_key===key);
  $('hint').textContent = spec.hint;
  $('slider').min = spec.min; $('slider').max = spec.max;
  $('slider').step = spec.decimals===1 ? 0.5 : 0.01;
  post({error_key:key});
}
errSel.onchange = () => setSpec(errSel.value);
$('side').onchange = () => post({target_side:$('side').value});
$('voice').onclick = () => { voiceOn=!voiceOn;
  $('voice').textContent = voiceOn?'🔊 語音提示: 開':'🔇 語音提示: 關';
  if(voiceOn) speechSynthesis.speak(new SpeechSynthesisUtterance(''));};
$('slider').oninput = e => { dragging=true;
  post({param:spec.param, value:parseFloat(e.target.value)}); };
$('slider').onchange = e => { dragging=false;
  post({param:spec.param, value:parseFloat(e.target.value), save:true}); };
$('reset').onclick = () => post({param:spec.param, reset:true, save:true});
$('copy').onclick = async () => {
  const t = await (await fetch('/snippet')).text();
  await navigator.clipboard.writeText(t); alert('已複製！貼回 HeuristicsConfig 或傳給 AI');
};
function post(obj){ fetch('/config',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)}); }
async function poll(){
  try{
    const s = await (await fetch('/status')).json();
    const v = s.metrics[spec.metric];
    $('big').textContent = v==null ? '--' :
        v.toFixed(spec.decimals)+spec.unit;
    const thr = s.config[spec.param];
    $('thr').textContent = '閾值 '+thr.toFixed(spec.decimals)+spec.unit+
        '（'+(spec.direction==='above'?'大於':'小於')+'觸發）預設 '+
        spec['default'].toFixed(spec.decimals);
    if(!dragging) $('slider').value = thr;
    const el = $('state');
    if(!s.person_visible){ el.className='ok'; el.textContent='偵測不到人'; }
    else if(s.triggered){ el.className='trig'; el.textContent='⚠ 會觸發 TRIGGERED'; }
    else { el.className='ok'; el.textContent='✓ 不觸發 OK  (fps '+s.fps.toFixed(0)+')'; }
    if(s.triggered && !lastTrig && voiceOn){
      speechSynthesis.speak(new SpeechSynthesisUtterance('觸發')); }
    lastTrig = s.triggered;
  }catch(e){}
  setTimeout(poll, 250);
}
setSpec('stance_too_high'); poll();
</script></body></html>"""


def build_app(state):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        specs = [dict(s, default=DEFAULT_CONFIG[s["param"]]) for s in SPECS]
        return PAGE.replace("%SPECS%", json.dumps(specs, ensure_ascii=False)) \
                   .replace("%%", "%")

    @app.get("/status")
    def status():
        with state.lock:
            return {
                "metrics": state.metrics, "config": state.config,
                "error_key": state.error_key, "triggered": state.triggered,
                "person_visible": state.person_visible, "fps": state.fps,
            }

    @app.post("/config")
    async def config(body: dict):
        with state.lock:
            if body.get("error_key") in SPEC_BY_KEY:
                state.error_key = body["error_key"]
            if body.get("target_side") in ("left", "right"):
                state.target_side = body["target_side"]
            param = body.get("param")
            if param in state.config:
                if body.get("reset"):
                    state.config[param] = DEFAULT_CONFIG[param]
                elif isinstance(body.get("value"), (int, float)):
                    state.config[param] = float(body["value"])
            if body.get("save"):
                state.save()
        return {"ok": True}

    @app.get("/snippet", response_class=PlainTextResponse)
    def snippet():
        with state.lock:
            return state.snippet()

    @app.get("/stream")
    def stream():
        def gen():
            while True:
                jpg = state.latest_jpeg
                if jpg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                time.sleep(0.07)
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


# ---------------------------------------------------------------------------
# Self-test: same synthetic en-garde as the Dart unit tests
# ---------------------------------------------------------------------------

def self_test():
    def stance(straight=False):
        knee = (0.56, 0.65) if straight else (0.62, 0.65)
        return {
            "left_hip": (0.48, 0.5), "right_hip": (0.52, 0.5),
            "right_knee": knee, "right_ankle": (0.60, 0.80),
            "left_ankle": (0.40, 0.80),
            "front_shoulder": (0.54, 0.35), "left_shoulder": (0.50, 0.35),
            "right_shoulder": (0.54, 0.35),
            "front_elbow": (0.64, 0.40), "front_wrist": (0.74, 0.45),
        }

    m = compute_window_metrics([stance()] * 40, 30, "left", DEFAULT_CONFIG)
    assert abs(m["avg_front_knee_angle_deg"] - 138.715) < 0.01, m
    assert abs(m["step_ratio_median"] - 2.0) < 1e-6, m
    assert abs(m["torso_lean_deg_median"] - 14.9314) < 0.01, m
    assert m["guard_below_pelvis_max_run_s"] == 0.0, m
    m2 = compute_window_metrics([stance(True)] * 40, 30, "left", DEFAULT_CONFIG)
    assert abs(m2["avg_front_knee_angle_deg"] - 180.0) < 0.01, m2

    # foot-before-hand: ankle rise onset at frame 5, wrist at 15 → 10/30 s
    seq = []
    for i in range(30):
        s = stance()
        if i >= 5:
            s["right_ankle"] = (0.65, 0.80)
        if i >= 15:
            s["front_wrist"] = (0.79, 0.45)
        seq.append(s)
    m3 = compute_window_metrics(seq, 30, "left", DEFAULT_CONFIG)
    assert abs(m3["foot_hand_lead_s"] - 10 / 30) < 1e-6, m3
    print("self-test OK — metrics match the Dart unit-test values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                        help="camera index (0) or a video file path")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    import socket
    import uvicorn

    state = TuningState()
    CameraLoop(state, args.source).start()

    hostname_ip = "?"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        hostname_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    print(f"\n📱 Phone: open  http://{hostname_ip}:{args.port}  (same Wi-Fi)\n")
    uvicorn.run(build_app(state), host="0.0.0.0", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()

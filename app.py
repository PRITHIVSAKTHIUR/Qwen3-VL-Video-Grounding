import colorsys
import gc
import tempfile
import re
import json
import uuid
import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from molmo_utils import process_vision_info

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID_V = "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX" # @--- Max model is trained on top of - Qwen/Qwen3-VL-4B-Instruct ---@
DTYPE = torch.bfloat16

print(f"Loading {MODEL_ID_V}...")
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V, trust_remote_code=True, torch_dtype=DTYPE
).to(device).eval()
print("Model loaded successfully.")

DEFAULT_MAX_SECONDS = 3.0
MAX_SECONDS_LIMIT = 20.0  

SYSTEM_PROMPT = """You are a helpful assistant to detect objects in images. When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] with the values being scaled between 0 and 1000. When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."""

POINT_SYSTEM_PROMPT = """You are a precise object pointing assistant. When asked to point to an object in an image, you must return ONLY the exact center coordinates of that specific object as [x, y] with values scaled between 0 and 1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner).

Rules:
1. ONLY point to objects that exactly match the description given.
2. Do NOT point to background, empty areas, or unrelated objects.
3. If there are multiple matching instances, return [[x1, y1], [x2, y2], ...].
4. If no matching object is found, return an empty list [].
5. Return ONLY the coordinate numbers, no explanations or other text.
6. Be extremely precise — place the point at the exact visual center of each matching object."""

POINTS_REGEX = re.compile(r'(?:(\d+)\s*[.:])?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)')
COORD_REGEX = re.compile(r'\[([\s\S]*?)\]')
FRAME_REGEX = re.compile(r'(\d+(?:\.\d+)?)\s*[,:]\s*([\d\s,\.]+)')

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )


def apply_gpu_duration(val: str):
    try:
        return int(val)
    except (TypeError, ValueError):
        return 300

def try_load_video_frames(video_path_or_url: str) -> tuple[list[Image.Image], dict]:
    cap = cv2.VideoCapture(video_path_or_url)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, {"num_frames": len(frames), "fps": float(fps_val) if fps_val > 0 else None}


def parse_bboxes_from_text(text: str) -> list[list[float]]:
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)
    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            all_b = []
            for m in nested:
                parsed = json.loads(m)
                all_b.extend(parsed if isinstance(parsed[0], list) else [parsed])
            return all_b
        except (json.JSONDecodeError, IndexError):
            pass
    single = re.findall(
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if single:
        return [[float(v) for v in m] for m in single]
    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    return [[float(nums[i]), float(nums[i + 1]), float(nums[i + 2]), float(nums[i + 3])] for i in
            range(0, len(nums) - 3, 4)] if len(nums) >= 4 else []


def parse_precise_points(text: str, image_w: int, image_h: int) -> list[tuple[float, float]]:
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)
    raw_points = []

    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            for m in nested:
                parsed = json.loads(m)
                if isinstance(parsed[0], list):
                    for p in parsed:
                        if len(p) >= 2:
                            raw_points.append((float(p[0]), float(p[1])))
                elif len(parsed) >= 2:
                    raw_points.append((float(parsed[0]), float(parsed[1])))
        except (json.JSONDecodeError, IndexError):
            pass

    if not raw_points:
        single = re.findall(
            r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
        if single:
            for m in single:
                raw_points.append((float(m[0]), float(m[1])))

    if not raw_points:
        for match in POINTS_REGEX.finditer(text):
            x_val = float(match.group(2))
            y_val = float(match.group(3))
            raw_points.append((x_val, y_val))

    validated = []
    for sx, sy in raw_points:
        if not (0 <= sx <= 1000 and 0 <= sy <= 1000):
            continue
        px = sx / 1000 * image_w
        py = sy / 1000 * image_h
        if 0 <= px <= image_w and 0 <= py <= image_h:
            validated.append((px, py))

    if len(validated) > 1:
        deduped = [validated[0]]
        for pt in validated[1:]:
            is_dup = False
            for existing in deduped:
                dist = ((pt[0] - existing[0]) ** 2 + (pt[1] - existing[1]) ** 2) ** 0.5
                if dist < 15:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(pt)
        validated = deduped

    return validated


def bbox_to_mask(bbox_scaled: list[float], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x1 = max(0, min(int(bbox_scaled[0] / 1000 * width), width - 1))
    y1 = max(0, min(int(bbox_scaled[1] / 1000 * height), height - 1))
    x2 = max(0, min(int(bbox_scaled[2] / 1000 * width), width - 1))
    y2 = max(0, min(int(bbox_scaled[3] / 1000 * height), height - 1))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def pixel_point_distance(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def overlay_masks_on_frame(frame: Image.Image, masks: dict, colors_map: dict, alpha=0.45) -> Image.Image:
    base = np.array(frame).astype(np.float32) / 255
    overlay = base.copy()
    for oid, mask in masks.items():
        if mask is None:
            continue
        color = np.array(colors_map.get(oid, (255, 0, 0)), dtype=np.float32) / 255
        m = np.clip(mask, 0, 1)[..., None]
        overlay = (1 - alpha * m) * overlay + (alpha * m) * color
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def pastel_color_for_prompt(prompt: str):
    hue = (sum(ord(c) for c in prompt) * 2654435761 % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)

class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames: list[Image.Image] = []
        self.video_fps: float | None = None
        self.masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        self.bboxes_by_frame: dict[int, dict[int, list[float]]] = {}
        self.color_by_obj: dict[int, tuple[int, int, int]] = {}
        self.color_by_prompt: dict[str, tuple[int, int, int]] = {}
        self.text_prompts_by_frame_obj: dict[int, dict[int, str]] = {}
        self.prompts: dict[str, list[int]] = {}
        self.next_obj_id: int = 1

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)


class PointTrackerState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames: list[Image.Image] = []
        self.video_fps: float | None = None
        self.points_by_frame: dict[int, list[tuple[float, float]]] = {}
        self.trails: list[list[tuple[int, float, float]]] = []

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)

def detect_objects_in_frame(frame: Image.Image, prompt: str) -> list[list[float]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame}, {"type": "text", "text": f"Detect all instances of: {prompt}"}]}
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_bboxes_from_text(txt)


def detect_precise_points_in_frame(frame: Image.Image, prompt: str) -> list[tuple[float, float]]:
    w, h = frame.size

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame},
                     {"type": "text",
                      "text": f"Detect all instances of: {prompt}. Return only bounding boxes for objects that exactly match this description."}]}
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]

    bboxes = parse_bboxes_from_text(txt)

    if bboxes:
        points = []
        for b in bboxes:
            bw = abs(b[2] - b[0])
            bh = abs(b[3] - b[1])
            if bw < 5 or bh < 5:
                continue
            if bw > 950 and bh > 950:
                continue
            cx = (b[0] + b[2]) / 2 / 1000 * w
            cy = (b[1] + b[3]) / 2 / 1000 * h
            if 0 <= cx <= w and 0 <= cy <= h:
                points.append((cx, cy))

        if len(points) > 1:
            deduped = [points[0]]
            for pt in points[1:]:
                is_dup = any(pixel_point_distance(pt, ex) < 20 for ex in deduped)
                if not is_dup:
                    deduped.append(pt)
            points = deduped

        if points:
            return points

    messages2 = [
        {"role": "system", "content": [{"type": "text", "text": POINT_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame},
                     {"type": "text",
                      "text": f"Point to the exact center of each '{prompt}' in this image. Only point to objects that are clearly '{prompt}', nothing else."}]}
    ]
    text2 = processor_v.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = processor_v(text=[text2], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model_v.generate(**inputs2, max_new_tokens=512, do_sample=False)
    generated2 = out2[:, inputs2.input_ids.shape[1]:]
    txt2 = processor_v.batch_decode(generated2, skip_special_tokens=True)[0]

    return parse_precise_points(txt2, w, h)

def track_prompt_across_frames(state: AppState, prompt: str):
    total = state.num_frames
    if prompt in state.prompts:
        for oid in state.prompts[prompt]:
            for f in range(total):
                state.masks_by_frame[f].pop(oid, None)
                state.bboxes_by_frame[f].pop(oid, None)
                state.text_prompts_by_frame_obj[f].pop(oid, None)
        del state.prompts[prompt]

    prev_tracks: list[tuple[int, list[float]]] = []

    for f_idx in range(total):
        frame = state.video_frames[f_idx]
        w, h = frame.size
        new_bboxes = detect_objects_in_frame(frame, prompt)

        masks_f = state.masks_by_frame.setdefault(f_idx, {})
        bboxes_f = state.bboxes_by_frame.setdefault(f_idx, {})
        texts_f = state.text_prompts_by_frame_obj.setdefault(f_idx, {})

        if not prev_tracks:
            for bbox in new_bboxes:
                oid = state.next_obj_id
                state.next_obj_id += 1
                if prompt not in state.color_by_prompt:
                    state.color_by_prompt[prompt] = pastel_color_for_prompt(prompt)
                state.color_by_obj[oid] = state.color_by_prompt[prompt]
                masks_f[oid] = bbox_to_mask(bbox, w, h)
                bboxes_f[oid] = bbox
                texts_f[oid] = prompt
                state.prompts.setdefault(prompt, []).append(oid)
                prev_tracks.append((oid, bbox))
            continue

        used = set()
        matched = {}
        scores = [(bbox_iou(pbbox, nbbox), pi, ni) for pi, (_, pbbox) in enumerate(prev_tracks) for ni, nbbox in
                  enumerate(new_bboxes)]
        scores.sort(reverse=True)
        for score, pi, ni in scores:
            if pi in matched or ni in used or score <= 0.05:
                continue
            matched[pi] = ni
            used.add(ni)

        for pi, (_, pbbox) in enumerate(prev_tracks):
            if pi in matched:
                continue
            best = min(((bbox_center_distance(pbbox, nbbox), ni) for ni, nbbox in enumerate(new_bboxes) if ni not in used),
                       default=(float('inf'), -1))
            if best[0] < 300:
                matched[pi] = best[1]
                used.add(best[1])

        new_prev = []
        for pi, (oid, _) in enumerate(prev_tracks):
            if pi in matched:
                nbbox = new_bboxes[matched[pi]]
                masks_f[oid] = bbox_to_mask(nbbox, w, h)
                bboxes_f[oid] = nbbox
                texts_f[oid] = prompt
                new_prev.append((oid, nbbox))
        for ni, nbbox in enumerate(new_bboxes):
            if ni not in used:
                oid = state.next_obj_id
                state.next_obj_id += 1
                if prompt not in state.color_by_prompt:
                    state.color_by_prompt[prompt] = pastel_color_for_prompt(prompt)
                state.color_by_obj[oid] = state.color_by_prompt[prompt]
                masks_f[oid] = bbox_to_mask(nbbox, w, h)
                bboxes_f[oid] = nbbox
                texts_f[oid] = prompt
                state.prompts.setdefault(prompt, []).append(oid)
                new_prev.append((oid, nbbox))
        prev_tracks = new_prev


def track_points_across_frames(pt_state: PointTrackerState, prompt: str):
    total = pt_state.num_frames
    prev_tracks: list[tuple[int, tuple[float, float]]] = []
    lost_count: dict[int, int] = {}

    for f_idx in range(total):
        frame = pt_state.video_frames[f_idx]
        w, h = frame.size

        new_points = detect_precise_points_in_frame(frame, prompt)
        points_f = pt_state.points_by_frame.setdefault(f_idx, [])

        if not prev_tracks:
            for px, py in new_points:
                track_idx = len(pt_state.trails)
                pt_state.trails.append([])
                points_f.append((px, py))
                pt_state.trails[track_idx].append((f_idx, px, py))
                prev_tracks.append((track_idx, (px, py)))
                lost_count[track_idx] = 0
            continue

        if not new_points:
            new_prev = []
            for track_idx, prev_pt in prev_tracks:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] > 5:
                    continue
                points_f.append(prev_pt)
                pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                new_prev.append((track_idx, prev_pt))
            prev_tracks = new_prev
            continue

        diag = (w ** 2 + h ** 2) ** 0.5
        match_threshold = diag * 0.25

        used_new = set()
        matched = {}

        dist_pairs = []
        for pi, (_, prev_pt) in enumerate(prev_tracks):
            for ni, new_pt in enumerate(new_points):
                d = pixel_point_distance(prev_pt, new_pt)
                dist_pairs.append((d, pi, ni))
        dist_pairs.sort()

        for d, pi, ni in dist_pairs:
            if pi in matched or ni in used_new:
                continue
            if d < match_threshold:
                matched[pi] = ni
                used_new.add(ni)

        new_prev = []
        for pi, (track_idx, prev_pt) in enumerate(prev_tracks):
            if pi in matched:
                ni = matched[pi]
                new_pt = new_points[ni]
                points_f.append(new_pt)
                pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                new_prev.append((track_idx, new_pt))
                lost_count[track_idx] = 0
            else:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] <= 5:
                    points_f.append(prev_pt)
                    pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                    new_prev.append((track_idx, prev_pt))

        for ni, new_pt in enumerate(new_points):
            if ni not in used_new:
                too_close = any(
                    pixel_point_distance(new_pt, prev_pt) < diag * 0.08
                    for _, prev_pt in new_prev
                )
                if not too_close:
                    track_idx = len(pt_state.trails)
                    pt_state.trails.append([])
                    points_f.append(new_pt)
                    pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                    new_prev.append((track_idx, new_pt))
                    lost_count[track_idx] = 0

        prev_tracks = new_prev


def render_point_tracker_video(pt_state: PointTrackerState, output_fps: int, trail_length: int = 12) -> str:
    RED = (255, 40, 40)
    DARK_RED = (180, 0, 0)
    frames_bgr = []

    for i in range(pt_state.num_frames):
        frame = pt_state.video_frames[i].copy()
        draw = ImageDraw.Draw(frame)

        points_f = pt_state.points_by_frame.get(i, [])

        for trail in pt_state.trails:
            trail_pts = [(tx, ty) for fi, tx, ty in trail if fi <= i and fi > i - trail_length]
            if len(trail_pts) >= 2:
                for t_idx in range(len(trail_pts) - 1):
                    alpha_ratio = (t_idx + 1) / len(trail_pts)
                    trail_color = (
                        int(DARK_RED[0] * alpha_ratio),
                        int(DARK_RED[1] * alpha_ratio),
                        int(DARK_RED[2] * alpha_ratio)
                    )
                    thickness = max(1, int(2 * alpha_ratio))
                    x1t, y1t = int(trail_pts[t_idx][0]), int(trail_pts[t_idx][1])
                    x2t, y2t = int(trail_pts[t_idx + 1][0]), int(trail_pts[t_idx + 1][1])
                    draw.line([(x1t, y1t), (x2t, y2t)], fill=trail_color, width=thickness)

        for (px, py) in points_f:
            r_outer = 10
            draw.ellipse(
                (px - r_outer, py - r_outer, px + r_outer, py + r_outer),
                outline="white", width=2
            )
            r = 7
            draw.ellipse(
                (px - r, py - r, px + r, py + r),
                fill=RED, outline=RED
            )
            r_inner = 2
            draw.ellipse(
                (px - r_inner, py - r_inner, px + r_inner, py + r_inner),
                fill=(255, 200, 200)
            )

        frames_bgr.append(np.array(frame)[:, :, ::-1])
        if (i + 1) % 30 == 0:
            gc.collect()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
            (frames_bgr[0].shape[1], frames_bgr[0].shape[0])
        )
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        return tmp.name


def render_full_video(state: AppState, output_fps: int) -> str:
    fps = output_fps
    frames_bgr = []
    for i in range(state.num_frames):
        frame = state.video_frames[i].copy()
        masks = state.masks_by_frame.get(i, {})
        if masks:
            frame = overlay_masks_on_frame(frame, masks, state.color_by_obj)
        bboxes = state.bboxes_by_frame.get(i, {})
        if bboxes:
            draw = ImageDraw.Draw(frame)
            w, h = frame.size
            for oid, bbox in bboxes.items():
                color = state.color_by_obj.get(oid, (255, 255, 255))
                x1 = int(bbox[0] / 1000 * w)
                y1 = int(bbox[1] / 1000 * h)
                x2 = int(bbox[2] / 1000 * w)
                y2 = int(bbox[3] / 1000 * h)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                prompt = state.text_prompts_by_frame_obj.get(i, {}).get(oid, "")
                if prompt:
                    label = f"{prompt} - ID{oid}"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                    except OSError:
                        font = ImageFont.load_default()
                    tb = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
                    draw.rectangle(tb, fill=color)
                    draw.text((x1 + 4, max(0, y1 - 27)), label, fill="white", font=font)
        frames_bgr.append(np.array(frame)[:, :, ::-1])
        if (i + 1) % 30 == 0:
            gc.collect()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (frames_bgr[0].shape[1], frames_bgr[0].shape[0]))
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        return tmp.name


def calc_gpu_duration_tracking(state, video, text_prompt, output_fps, max_seconds, gpu_timeout):  # Changed
    try:
        return int(gpu_timeout)
    except (TypeError, ValueError):
        return 300


def calc_gpu_duration_points(pt_state, video, text_prompt, output_fps, max_seconds, gpu_timeout):  # Changed
    try:
        return int(gpu_timeout)
    except (TypeError, ValueError):
        return 300


def calc_gpu_duration_qa(video, user_text, max_new_tokens, gpu_timeout):
    try:
        return int(gpu_timeout)
    except (TypeError, ValueError):
        return 300


@spaces.GPU(duration=calc_gpu_duration_tracking)
def process_and_render(state: AppState, video, text_prompt: str, output_fps: int, max_seconds: float, gpu_timeout: int):  # Changed
    if video is None:
        return "❌ Please upload a video", None
    if not text_prompt or not text_prompt.strip():
        return "❌ Please enter at least one text prompt", None

    state.reset()
    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video
    frames, info = try_load_video_frames(path)
    if not frames:
        return "❌ Could not load video", None
    if info["fps"] and len(frames) > max_seconds * info["fps"]:  # Changed
        frames = frames[:int(max_seconds * info["fps"])]  # Changed
    state.video_frames = frames
    state.video_fps = info["fps"]

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    status = f"✅ Video loaded: {state.num_frames} frames\n"
    status += f"Output FPS: {output_fps}\n"
    status += f"Max Seconds: {max_seconds}s\n"  # Changed
    status += f"GPU Duration: {gpu_timeout}s\n"
    status += f"Processing {len(prompts)} prompt(s) across ALL frames...\n\n"

    for p in prompts:
        track_prompt_across_frames(state, p)
        count = len(state.prompts.get(p, []))
        status += f"• '{p}': {count} object(s) tracked\n"

    status += "\n🎥 Rendering final video with overlays..."
    rendered_path = render_full_video(state, output_fps)
    status += "\n\n✅ Done! Play the video below."

    return status, rendered_path


@spaces.GPU(duration=calc_gpu_duration_points)
def process_and_render_points(pt_state: PointTrackerState, video, text_prompt: str, output_fps: int, max_seconds: float, gpu_timeout: int):  # Changed
    if video is None:
        return "❌ Please upload a video", None
    if not text_prompt or not text_prompt.strip():
        return "❌ Please enter at least one text prompt", None

    pt_state.reset()
    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video
    frames, info = try_load_video_frames(path)
    if not frames:
        return "❌ Could not load video", None
    if info["fps"] and len(frames) > max_seconds * info["fps"]:  # Changed
        frames = frames[:int(max_seconds * info["fps"])]  # Changed
    pt_state.video_frames = frames
    pt_state.video_fps = info["fps"]

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    status = f"✅ Video loaded: {pt_state.num_frames} frames\n"
    status += f"Output FPS: {output_fps}\n"
    status += f"Max Seconds: {max_seconds}s\n"  # Changed
    status += f"GPU Duration: {gpu_timeout}s\n"
    status += f"Processing {len(prompts)} prompt(s) with point tracking...\n\n"

    for p in prompts:
        track_points_across_frames(pt_state, p)
        status += f"• '{p}': tracked\n"

    total_tracked = len(pt_state.trails)
    status += f"\n📍 Total tracked points: {total_tracked}\n"
    status += "\n🎥 Rendering video with red dot tracking..."
    rendered_path = render_point_tracker_video(pt_state, output_fps)
    status += "\n\n✅ Done! Play the video below."

    return status, rendered_path


@spaces.GPU(duration=calc_gpu_duration_qa)
def process_video_qa(video, user_text, max_new_tokens, gpu_timeout):
    if video is None:
        return "❌ Please upload a video."

    if not user_text or not user_text.strip():
        user_text = "Describe this video in detail."

    if isinstance(video, dict):
        video_path = video.get("name") or video.get("path") or video.get("data")
    else:
        video_path = video

    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=user_text),
                dict(type="video", video=video_path),
            ],
        }
    ]

    try:
        _, videos, video_kwargs = process_vision_info(messages)
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    except Exception as e:
        return f"❌ Error processing video frames: {e}"

    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor_v(
        videos=videos,
        video_metadata=video_metadatas,
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model_v.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model_v.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor_v.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    generated_text = re.sub(r'<think>.*?</think>', '', generated_text.strip(), flags=re.DOTALL).strip()

    return generated_text


css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
}
#main-title h1 {font-size: 2.6em !important;}

/* RadioAnimated Styles */
.ra-wrap{ width: fit-content; }
.ra-inner{
  position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
  background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
}
.ra-input{ display: none; }
.ra-label{
  position: relative; z-index: 2; padding: 8px 16px;
  font-family: inherit; font-size: 14px; font-weight: 600;
  color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
}
.ra-highlight{
  position: absolute; z-index: 1; top: 6px; left: 6px;
  height: calc(100% - 12px); border-radius: 9999px;
  background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s, width 0.2s;
}
.ra-input:checked + .ra-label{ color: black; }

/* Dark mode adjustments for RadioAnimated */
.dark .ra-inner { background: var(--neutral-800); }
.dark .ra-label { color: var(--neutral-400); }
.dark .ra-highlight { background: var(--neutral-600); }
.dark .ra-input:checked + .ra-label { color: white; }

#gpu-duration-container {
    padding: 16px;
    border-radius: 12px;
    background: var(--background-fill-secondary);
    border: 2px solid var(--border-color-primary);
    margin-top: 8px;
}

#gpu-info-box {
    padding: 12px;
    border-radius: 8px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-secondary);
}
"""


with gr.Blocks() as demo:
    gr.Markdown("# **Qwen3-VL-Video-Grounding**", elem_id="main-title")

    gr.Markdown(
        """
        Perform point tracking, text-guided detection, and video question answering with the Qwen3-VL multimodal model. This demo runs the official implementation using the Hugging Face Transformers, OpenCV, and Molmo libraries. <span style="color: orange;">Note: This app will consume more ZeroGPU compute and may sometimes end up producing no results. Please replicate the app and try it in your environment for better usage.</span>
        """
    )  # Changed
    
    state = gr.State(AppState())
    pt_state = gr.State(PointTrackerState())
    gpu_duration_state = gr.State(value=300)

    with gr.Tabs():

        with gr.Tab("Text-guided Object Tracking"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Getting started**
                        - **Upload a video** or record from webcam.
                        - Enter **object descriptions** separated by commas (e.g. `person, red car, dog`).
                        - Each prompt can detect **multiple instances(classes)** — they'll each get a unique filter **ID's**.
                        - Adjust **Max Video Seconds** slider to control how much of the video is processed.
                        """
                    )  # Changed
                with gr.Column():
                    gr.Markdown(
                        """
                        **How tracking works**
                        - The model detects **bounding boxes** for each object in every frame.
                        - Objects are matched across frames using **IoU overlap** and **center-distance** tracking.
                        - Output includes colored bounding boxes, semi-transparent mask overlays, and labeled IDs.
                        """
                    )

            with gr.Column():
                with gr.Row():
                    video_in = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)

                with gr.Row():
                    prompt_in = gr.Textbox(
                        label="Text Prompts (comma separated)",
                        placeholder="person, red car, dog, laptop, traffic light",
                        lines=3
                    )
                with gr.Row():
                    fps_slider = gr.Slider(
                        label="Output Video FPS",
                        minimum=1,
                        maximum=60,
                        value=25,
                        step=1,
                        info="Default: 25 FPS (BEST)"
                    )
                with gr.Row():  # Changed
                    max_seconds_slider = gr.Slider(  # Changed
                        label="Max Video Seconds",  # Changed
                        minimum=1,  # Changed
                        maximum=MAX_SECONDS_LIMIT,  # Changed
                        value=DEFAULT_MAX_SECONDS, 
                        step=1,  # Changed
                        info=f"Default: {int(DEFAULT_MAX_SECONDS)}s — Max: {int(MAX_SECONDS_LIMIT)}s. Longer = more frames = slower processing."  # Changed
                    )  # Changed

            process_btn = gr.Button("Apply Detection and Render Full Video", variant="primary")

            status_out = gr.Textbox(label="Output Status", lines=3)
            rendered_out = gr.Video(label="Rendered Video with Object Tracking", height=400)

            gr.Examples(
                examples=[
                    ["examples/1.mp4"],
                    ["examples/2.mp4"],
                    ["examples/3.mp4"],
                ],
                inputs=[video_in],
                label="Examples"
            )

        with gr.Tab("Points Tracker"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Getting started**
                        - **Upload a video** or record from webcam.
                        - Enter **object descriptions** separated by commas (e.g. `person, ball, face`).
                        - The model locates the **center point** of each detected object and tracks it with a **red dot**.
                        - Adjust **Max Video Seconds** slider to control how much of the video is processed.
                        """
                    )  # Changed
                with gr.Column():
                    gr.Markdown(
                        """
                        **How point tracking works**
                        - Uses **bounding box detection** converted to precise **center points** for reliability.
                        - Points are matched across frames using **adaptive nearest-neighbor** tracking.
                        - Lost tracks are kept for up to 5 frames, then dropped to avoid ghost points.
                        - Clean visualization with **red dots** and subtle **motion trails**.
                        """
                    )

            with gr.Column():
                with gr.Row():
                    pt_video_in = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)

                with gr.Row():
                    pt_prompt_in = gr.Textbox(
                        label="Text Prompts (comma separated)",
                        placeholder="person, ball, car, face, hand",
                        lines=3
                    )
                with gr.Row():
                    pt_fps_slider = gr.Slider(
                        label="Output Video FPS",
                        minimum=1,
                        maximum=60,
                        value=25,
                        step=1,
                        info="Default: 25 FPS (BEST)"
                    )
                with gr.Row():  # Changed
                    pt_max_seconds_slider = gr.Slider(  # Changed
                        label="Max Video Seconds",  # Changed
                        minimum=1,  # Changed
                        maximum=MAX_SECONDS_LIMIT,  # Changed
                        value=DEFAULT_MAX_SECONDS,  # Changed
                        step=1,  # Changed
                        info=f"Default: {int(DEFAULT_MAX_SECONDS)}s — Max: {int(MAX_SECONDS_LIMIT)}s. Longer = more frames = slower processing."  # Changed
                    )  # Changed

            pt_process_btn = gr.Button("Apply Point Tracking & Render Video", variant="primary")

            pt_status_out = gr.Textbox(label="Output Status", lines=5)
            pt_rendered_out = gr.Video(label="Rendered Video with Point Tracking", height=400)

            gr.Examples(
                examples=[
                    ["examples/1.mp4"],
                    ["examples/2.mp4"],
                    ["examples/3.mp4"],
                ],
                inputs=[pt_video_in],
                label="Examples"
            )
        
        with gr.Tab("Any Video QA"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Getting started**
                        - **Upload a video** or record from webcam.
                        - Enter a **question or prompt** about the video content.
                        - The model will analyze the video and provide a **text answer**.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **How it works**
                        - The video frames are processed by the **Qwen3-VL** vision-language model.
                        - You can ask **any question** about the video: describe scenes, identify actions, count objects, etc.
                        - If no prompt is provided, the model will **describe the video in detail**.
                        """
                    )

            with gr.Column():
                with gr.Row():
                    qa_video_in = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)

                with gr.Row():
                    qa_prompt_in = gr.Textbox(
                        label="Text Prompt / Question",
                        placeholder="Describe this video in detail. / What is happening in this video? / How many people are visible?",
                        lines=3
                    )
                with gr.Row():
                    qa_max_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=64,
                        maximum=2048,
                        value=1024,
                        step=64,
                        info="Maximum number of tokens in the generated response"
                    )

            qa_process_btn = gr.Button("Analyze Video", variant="primary")

            qa_output = gr.Textbox(label="Model Response", lines=12)

            gr.Examples(
                examples=[
                    ["examples/1.mp4"],
                    ["examples/2.mp4"],
                    ["examples/3.mp4"],
                ],
                inputs=[qa_video_in],
                label="Examples"
            )
        
        with gr.Tab("ZeroGPU Duration(Default=300)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ## ZeroGPU Duration Settings

                        Configure the **maximum GPU allocation time** for all processing tasks across every tab.
                        This setting is **shared globally** — changing it here affects:

                        -  **Text-guided Object Tracking** (Tab 1)
                        -  **Points Tracker** (Tab 2)
                        -  **Any Video QA** (Tab 3)
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        f"""
                        ## Duration Guide

                        | Duration | Best For |
                        |----------|----------|
                        | **60s** | Short videos (1-3s), simple prompts |
                        | **120s** | Medium videos (3-5s), 1-2 prompts |
                        | **180s** | Longer videos (5-10s), multiple prompts |
                        | **240s** | Long videos (10-15s), complex multi-object tracking |
                        | **300s+** | Maximum length videos (up to {int(MAX_SECONDS_LIMIT)}s) |
                        """
                    )  # Changed

            with gr.Column():
                with gr.Row(elem_id="gpu-duration-container"):
                    with gr.Column():
                        gr.Markdown("### Select GPU Duration (seconds)")
                        gr.Markdown(
                            "*Slide to choose how long the GPU will be reserved for each processing request. "
                            "Higher values allow longer/more complex videos but consume more GPU quota.*"
                        )
                        radioanimated_gpu_duration = RadioAnimated(
                            choices=["60", "90", "120", "180", "240", "300", "360"],
                            value="300",
                            elem_id="radioanimated_gpu_duration"
                        )

                with gr.Row():
                    with gr.Column(elem_id="gpu-info-box"):
                        gpu_display = gr.Markdown(
                            value="**Currently selected:** `300 seconds`"
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            f"""
                            ### Important Notes

                            - **Higher duration = more GPU quota consumed.** Choose the minimum needed for your task.
                            - On **Hugging Face ZeroGPU Spaces**, each user has a daily GPU quota. Be mindful of usage.
                            - If processing **times out**, increase the duration and retry.
                            - The duration is the **maximum allowed time** — if processing finishes early, the GPU is released.
                            - **Default: 300 seconds** is sufficient for most short video tasks.
                            - **Max Video Seconds** (up to {int(MAX_SECONDS_LIMIT)}s) can be adjusted per-tab via the slider.
                            """
                        )  # Changed

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            f"""
                            ### 🔧 Troubleshooting

                            | Issue | Solution |
                            |-------|----------|
                            | Processing times out | Increase GPU duration to 240s or 300s |
                            | GPU quota exhausted | Wait for quota reset or use shorter durations |
                            | Video too long | Reduce Max Video Seconds slider (default: {int(DEFAULT_MAX_SECONDS)}s, max: {int(MAX_SECONDS_LIMIT)}s) |
                            | Multiple prompts slow | Use fewer comma-separated prompts or increase duration |
                            """
                        )  # Changed

    def update_gpu_display(val: str):
        duration = apply_gpu_duration(val)
        return duration, f"**Currently selected:** `{duration} seconds`"

    radioanimated_gpu_duration.change(
        fn=update_gpu_display,
        inputs=radioanimated_gpu_duration,
        outputs=[gpu_duration_state, gpu_display],
        api_visibility="private"
    )
    
    process_btn.click(
        fn=process_and_render,
        inputs=[state, video_in, prompt_in, fps_slider, max_seconds_slider, gpu_duration_state],  # Changed
        outputs=[status_out, rendered_out],
        show_progress=True
    )

    pt_process_btn.click(
        fn=process_and_render_points,
        inputs=[pt_state, pt_video_in, pt_prompt_in, pt_fps_slider, pt_max_seconds_slider, gpu_duration_state],  # Changed
        outputs=[pt_status_out, pt_rendered_out],
        show_progress=True
    )

    qa_process_btn.click(
        fn=process_video_qa,
        inputs=[qa_video_in, qa_prompt_in, qa_max_tokens, gpu_duration_state],
        outputs=[qa_output],
        show_progress=True
    )

demo.queue().launch(css=css, theme=Soft(primary_hue="orange", secondary_hue="rose"), ssr_mode=False, mcp_server=True)
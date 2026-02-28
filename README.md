# **Qwen3-VL-Video-Grounding**

A Gradio-based web application for video object tracking, point tracking, and video question answering using the Qwen3-VL multimodal vision-language model. Supports text-guided detection with bounding box overlays, precise point tracking with motion trails, and open-ended video comprehension through natural language queries.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Tab 1: Text-guided Object Tracking](#tab-1-text-guided-object-tracking)
- [Tab 2: Points Tracker](#tab-2-points-tracker)
- [Tab 3: Any Video QA](#tab-3-any-video-qa)
- [Tab 4: ZeroGPU Duration Settings](#tab-4-zerogpu-duration-settings)
- [Configuration Parameters](#configuration-parameters)
- [Technical Details](#technical-details)
- [Object Tracking Algorithm](#object-tracking-algorithm)
- [Point Tracking Algorithm](#point-tracking-algorithm)
- [Video Processing Pipeline](#video-processing-pipeline)
- [System Prompts](#system-prompts)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [License](#license)
- [Credits](#credits)

## Overview

Qwen3-VL-Video-Grounding is a multi-functional video analysis tool built on top of the Qwen3-VL vision-language model. It processes video frames individually through the model to detect, locate, and track objects across time. The application provides three primary capabilities:

1. **Text-guided Object Tracking** -- Detects objects matching text descriptions and tracks them across frames with colored bounding boxes and semi-transparent mask overlays.
2. **Point Tracking** -- Locates the precise center point of described objects and tracks them with red dot markers and motion trails.
3. **Video Question Answering** -- Accepts natural language questions about video content and generates detailed text responses.

All processing runs client-side through a Gradio interface with ZeroGPU support for Hugging Face Spaces deployment.

## Features

- **Multi-Object Detection** -- Detect multiple instances of different object classes simultaneously using comma-separated text prompts.
- **Cross-Frame Tracking** -- Objects are matched across frames using IoU overlap and center-distance metrics to maintain consistent identity assignments.
- **Bounding Box Visualization** -- Colored bounding boxes with semi-transparent mask overlays and labeled IDs rendered onto output video.
- **Precise Point Tracking** -- Center-point localization with red dot markers, white outlines, and fading motion trails.
- **Video QA** -- Open-ended question answering about video content using the full vision-language model capabilities.
- **Configurable GPU Duration** -- Adjustable ZeroGPU allocation time shared across all processing tabs.
- **Max Video Seconds Control** -- Per-tab slider to limit how much of a video is processed (default 3 seconds, maximum 20 seconds).
- **Webcam Support** -- Record directly from webcam in addition to file upload.
- **Custom Theme** -- Orange and rose themed interface built on Gradio's Soft theme.
- **MCP Server Support** -- Launches with Model Context Protocol server enabled.
- **Memory Management** -- Periodic garbage collection during frame rendering to manage memory usage.

## Model Details

| Component | Value |
|-----------|-------|
| Model ID | [prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX](https://huggingface.co/prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX) |
| Base Architecture | Qwen3-VL (trained on top of Qwen/Qwen3-VL-4B-Instruct) |
| Model Class | Qwen3VLForConditionalGeneration |
| Precision | bfloat16 |
| Processor | AutoProcessor with trust_remote_code |
| Inference Mode | Evaluation (model.eval()) |

## Requirements

### Hardware

- CUDA-capable GPU with sufficient VRAM (recommended 16GB or more)
- CPU fallback is supported but not practical for video processing

### Software

- Python 3.8 or higher
- PyTorch with CUDA support
- Gradio
- OpenCV (cv2)
- NumPy
- Pillow
- Transformers (Hugging Face)

### Python Dependencies

```
torch
gradio
numpy
Pillow
opencv-python
transformers
spaces
```

### Additional Modules

The application requires a local `molmo_utils` module providing the `process_vision_info` function for video frame extraction in the QA pipeline.

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Qwen3-VL-Video-Grounding
```

### Install Dependencies

```bash
pip install torch torchvision gradio numpy Pillow opencv-python transformers
pip install spaces
```

### Model Download

The model is automatically downloaded from HuggingFace on first run:

```
prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX
```

Ensure sufficient disk space and network access to HuggingFace.

### Example Videos

Place example video files in an `examples/` directory:

```
examples/
  1.mp4
  2.mp4
  3.mp4
```

### Launch

```bash
python app.py
```

The application starts and displays a local URL (typically `http://127.0.0.1:7860`).

## Usage

The application is organized into four tabs, each serving a distinct purpose.

## Tab 1: Text-guided Object Tracking

### Purpose

Detects objects matching text descriptions in every frame of a video and tracks them with colored bounding boxes, semi-transparent mask overlays, and labeled identifiers.

### Workflow

1. Upload a video or record from webcam.
2. Enter one or more object descriptions separated by commas (for example, `person, red car, dog`).
3. Adjust Output Video FPS (default: 25).
4. Adjust Max Video Seconds (default: 3, maximum: 20).
5. Click **Apply Detection and Render Full Video**.
6. View the status log and rendered output video.

### How It Works

- The model detects bounding boxes for each described object in every frame.
- Objects are matched across frames using IoU overlap and center-distance tracking.
- Each unique prompt receives a consistent pastel color generated from a hash of the prompt string.
- Each detected instance receives a unique numeric ID that persists across frames.
- Output includes colored bounding boxes (4px width), semi-transparent mask overlays (45% opacity), and text labels showing the prompt and ID.

### Output Format

- Rendered MP4 video with overlaid detections
- Status text showing frame count, FPS, processing details, and object counts per prompt

## Tab 2: Points Tracker

### Purpose

Locates the precise center point of described objects and tracks them across frames with red dot markers and fading motion trails.

### Workflow

1. Upload a video or record from webcam.
2. Enter one or more object descriptions separated by commas (for example, `person, ball, face`).
3. Adjust Output Video FPS (default: 25).
4. Adjust Max Video Seconds (default: 3, maximum: 20).
5. Click **Apply Point Tracking & Render Video**.
6. View the status log and rendered output video.

### How It Works

- Uses bounding box detection converted to precise center points for reliability.
- A fallback point-specific prompt is used if bounding box detection fails.
- Points are matched across frames using adaptive nearest-neighbor tracking with a distance threshold of 25% of the frame diagonal.
- Lost tracks are preserved for up to 5 frames before being dropped to prevent ghost points.
- Duplicate points closer than 20 pixels (detection) or 8% of the diagonal (new tracks) are filtered out.

### Visual Elements

| Element | Description |
|---------|-------------|
| Red dot | 7px radius filled circle at the detected center point |
| White outline | 10px radius outer ring around each dot |
| Inner highlight | 2px radius light pink center dot |
| Motion trail | Fading dark red lines connecting previous positions (up to 12 frames) |

## Tab 3: Any Video QA

### Purpose

Accepts natural language questions about video content and generates detailed text responses using the full vision-language model.

### Workflow

1. Upload a video or record from webcam.
2. Enter a question or prompt about the video (for example, `What is happening in this video?`).
3. Adjust Max New Tokens (default: 1024, range: 64 to 2048).
4. Click **Analyze Video**.
5. Read the generated text response.

### How It Works

- Video frames are extracted and processed using the `process_vision_info` utility from the `molmo_utils` module.
- The frames and text prompt are passed together to the Qwen3-VL model.
- If no prompt is provided, the default prompt "Describe this video in detail." is used.
- Any `<think>` tags in the model output are stripped from the final response.
- Generation uses `torch.inference_mode()` for optimal performance.

### Supported Question Types

- Scene description
- Action identification
- Object counting
- Event sequencing
- Spatial relationship queries
- Any open-ended question about the video content

## Tab 4: ZeroGPU Duration Settings

### Purpose

Configures the maximum GPU allocation time for all processing tasks across every tab. This is a global setting that affects Tab 1, Tab 2, and Tab 3.

### Available Durations

| Duration (seconds) | Best For |
|--------------------|----------|
| 60 | Short videos (1-3s), simple prompts |
| 90 | Short videos with multiple prompts |
| 120 | Medium videos (3-5s), 1-2 prompts |
| 180 | Longer videos (5-10s), multiple prompts |
| 240 | Long videos (10-15s), complex multi-object tracking |
| 300 | Maximum length videos (up to 20s) |
| 360 | Extended processing for complex scenarios |

### Default Value

300 seconds. This is sufficient for most short video tasks.

### Notes

- Higher duration consumes more GPU quota on Hugging Face ZeroGPU Spaces.
- The duration is the maximum allowed time. If processing finishes early, the GPU is released.
- If processing times out, increase the duration and retry.

## Configuration Parameters

### Global Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DEFAULT_MAX_SECONDS` | 3.0 | Default maximum video duration to process |
| `MAX_SECONDS_LIMIT` | 20.0 | Absolute maximum video duration allowed |
| `DTYPE` | bfloat16 | Model precision |

### Per-Tab Parameters

#### Object Tracking and Point Tracking

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Output Video FPS | 1 to 60 | 25 | Frame rate of the rendered output video |
| Max Video Seconds | 1 to 20 | 3 | Maximum seconds of input video to process |

#### Video QA

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Max New Tokens | 64 to 2048 | 1024 | Maximum tokens in the generated text response |

## Technical Details

### Object Tracking Algorithm

The cross-frame tracking algorithm operates as follows:

1. **Detection** -- For each frame, the model generates bounding boxes in normalized coordinates (0-1000 scale) for each text prompt.
2. **Initial Frame** -- All detected bounding boxes become new tracked objects with unique IDs.
3. **Subsequent Frames** -- A matching process assigns new detections to existing tracks:
   - Compute IoU (Intersection over Union) between all pairs of previous and new bounding boxes.
   - Sort by IoU descending and greedily assign matches with IoU greater than 0.05.
   - For unmatched previous tracks, fall back to center-distance matching with a threshold of 300 units (in 1000-scale coordinates).
4. **New Objects** -- Unmatched new detections become new tracked objects with fresh IDs.
5. **Color Assignment** -- Each unique text prompt receives a consistent pastel color generated by hashing the prompt string through HSV color space (saturation 0.5, value 0.95).

### Point Tracking Algorithm

The point tracking algorithm uses a two-stage detection approach:

1. **Primary Detection** -- Bounding box detection via the standard system prompt, converted to center points.
   - Boxes smaller than 5 units or larger than 950 units (in 1000-scale) are filtered out.
   - Points closer than 20 pixels are deduplicated.
2. **Fallback Detection** -- If no valid points are found from bounding boxes, a dedicated point-detection prompt is used.
3. **Cross-Frame Matching** -- Nearest-neighbor matching with adaptive threshold (25% of frame diagonal).
   - Lost tracks are preserved for up to 5 consecutive frames.
   - New points too close to existing tracks (within 8% of diagonal) are suppressed.
   - Tracks exceeding the lost frame limit are permanently dropped.

### Bounding Box Parsing

The response parser handles multiple formats:

1. **Nested arrays** -- `[[x1,y1,x2,y2], [x1,y1,x2,y2]]`
2. **Single arrays** -- `[x1, y1, x2, y2]`
3. **Raw numbers** -- Sequences of four numbers extracted via regex

All coordinates use a 0-1000 normalized scale where (0,0) is top-left and (1000,1000) is bottom-right.

### Point Coordinate Parsing

Point responses are parsed through multiple strategies:

1. **Nested arrays** -- `[[x1,y1], [x2,y2]]`
2. **Single arrays** -- `[x, y]`
3. **Regex pattern** -- `number, number` with optional index prefix

Points are validated to be within the 0-1000 range and deduplicated with a 15-pixel minimum distance.

### Video Processing Pipeline

1. **Frame Extraction** -- OpenCV reads the video file and converts each frame from BGR to RGB as PIL Images.
2. **Frame Limiting** -- If the video exceeds the max seconds setting, frames beyond the limit are discarded based on the detected FPS.
3. **Per-Frame Processing** -- Each frame is sent to the model independently for detection.
4. **Rendering** -- Detected elements are drawn onto copies of the original frames using Pillow's ImageDraw.
5. **Video Encoding** -- Rendered frames are written to a temporary MP4 file using OpenCV's VideoWriter with the mp4v codec.
6. **Memory Management** -- `gc.collect()` is called every 30 frames during rendering.

### Mask Overlay Rendering

For object tracking, masks are rendered as semi-transparent colored overlays:

```
overlay = (1 - alpha * mask) * base + (alpha * mask) * color
```

Where alpha is 0.45 (45% opacity).

## System Prompts

### Bounding Box Detection Prompt

```
You are a helpful assistant to detect objects in images. When asked to detect 
elements based on a description you return bounding boxes for all elements in 
the form of [xmin, ymin, xmax, ymax] with the values being scaled between 0 
and 1000. When there are more than one result, answer with a list of bounding 
boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...].
```

### Point Detection Prompt (Fallback)

```
You are a precise object pointing assistant. When asked to point to an object 
in an image, you must return ONLY the exact center coordinates of that specific 
object as [x, y] with values scaled between 0 and 1000 (where 0,0 is the 
top-left corner and 1000,1000 is the bottom-right corner).

Rules:
1. ONLY point to objects that exactly match the description given.
2. Do NOT point to background, empty areas, or unrelated objects.
3. If there are multiple matching instances, return [[x1, y1], [x2, y2], ...].
4. If no matching object is found, return an empty list [].
5. Return ONLY the coordinate numbers, no explanations or other text.
6. Be extremely precise — place the point at the exact visual center of each 
   matching object.
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Processing times out | GPU duration too short for video length | Increase GPU duration to 240s or 300s in Tab 4 |
| GPU quota exhausted | Too many processing requests | Wait for quota reset or use shorter durations |
| Video too long | Exceeds max seconds setting | Reduce the Max Video Seconds slider |
| Multiple prompts slow | Each prompt requires per-frame inference | Use fewer comma-separated prompts or increase GPU duration |
| No objects detected | Prompt too vague or objects too small | Use more specific descriptions; ensure objects are visible |
| Ghost points in tracker | Lost track threshold too high | Points are automatically dropped after 5 lost frames |
| No results produced | ZeroGPU compute limitations | Replicate the app and run in a dedicated environment |
| Video QA returns empty | Model could not parse video frames | Ensure the video is in a supported format (MP4 recommended) |
| Font rendering error | DejaVuSans font not found on system | Falls back to Pillow's default font automatically |

## Limitations

- **Per-Frame Detection** -- Each frame is processed independently by the model, which is computationally expensive. Processing time scales linearly with frame count and number of prompts.
- **No Optical Flow** -- The tracking algorithm relies on detection consistency rather than optical flow or feature matching, which may cause ID switches during occlusion or rapid movement.
- **Maximum Video Duration** -- Videos are limited to 20 seconds maximum to manage GPU memory and processing time.
- **Fixed Negative Filtering** -- Point tracker filters out bounding boxes smaller than 5 units or larger than 950 units, which may miss very small or full-frame objects.
- **Lost Track Recovery** -- Once a point track is lost for more than 5 consecutive frames, it is permanently dropped and cannot be recovered.
- **Single Model** -- All three capabilities (tracking, pointing, QA) use the same model, so only one task can run at a time.
- **Think Tag Stripping** -- The model may generate reasoning in `<think>` tags which are stripped from output. This processing is done via regex and may fail on malformed tags.
- **MP4V Codec** -- Output videos use the mp4v codec which may have compatibility issues with some players. Re-encoding with H.264 may be necessary for broader compatibility.
- **ZeroGPU Compute** -- On Hugging Face Spaces, the application consumes significant GPU compute and may sometimes produce no results due to resource constraints.

## Application Architecture

### State Management

The application uses two state classes:

**AppState** (Object Tracking)
```
- video_frames: list of PIL Images
- video_fps: detected FPS from input video
- masks_by_frame: frame index -> object ID -> numpy mask array
- bboxes_by_frame: frame index -> object ID -> [xmin, ymin, xmax, ymax]
- color_by_obj: object ID -> (R, G, B) tuple
- color_by_prompt: prompt string -> (R, G, B) tuple
- text_prompts_by_frame_obj: frame index -> object ID -> prompt string
- prompts: prompt string -> list of object IDs
- next_obj_id: auto-incrementing integer
```

**PointTrackerState** (Point Tracking)
```
- video_frames: list of PIL Images
- video_fps: detected FPS from input video
- points_by_frame: frame index -> list of (x, y) tuples
- trails: list of trail sequences, each containing (frame_index, x, y) tuples
```

### Custom UI Component

The application includes a custom `RadioAnimated` component extending `gr.HTML` that provides an animated pill-style radio button selector. This component is used for GPU duration selection in Tab 4 and supports both light and dark mode themes.

### Server Configuration

| Setting | Value |
|---------|-------|
| Queue | Enabled (default settings) |
| Theme | Soft (primary: orange, secondary: rose) |
| SSR Mode | Disabled |
| MCP Server | Enabled |

## License

This project uses models and libraries from HuggingFace. Refer to the individual model and library repositories for their respective licenses:

- [prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX](https://huggingface.co/prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX)
- [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

"""nature_scene_agent – Graph node that classifies whether an image is a
nature / outdoor shot using a local Ollama vision model.

Responsibilities
-----------------
- For every candidate image, send it to Ollama's vision model.
- Ask the model to classify the scene as nature/outdoor or not.
- Parse the response into a ``NatureSceneResult``.
- Supplement with fast OpenCV colour-histogram ratios (green/blue/earth).

Performance note
----------------
Vision-model inference is the slowest step.  This node **pre-filters** by
reading ``quality_results`` and only processing images that are sharp and
clean (passed the quality gate).  Images that failed quality get a default
``NatureSceneResult`` with ``is_nature=False``.

Memory safety
-------------
Each image is loaded, analysed, and immediately freed before moving
to the next one.  Base64-encoded data is deleted right after the
Ollama call.  Explicit ``gc.collect()`` after each image prevents
memory accumulation.
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from photo_selector.agents.state import (
    NatureSceneResult,
    PhotoSelectorState,
    QualityResult,
)

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VISION_MODEL = "llama3.2-vision"
DEFAULT_BASE_URL = "http://localhost:11434"

NATURE_PROMPT = """\
Look at this image carefully.

1. Is this a nature or outdoor scene? (yes / no)
2. Rate your confidence that this is a nature scene from 0.0 to 1.0.
3. List the nature-related elements you see (e.g. trees, sky, water, mountains, flowers, grass, forest, lake, sunset).

Return ONLY a JSON object with these keys:
  "is_nature": true or false,
  "confidence": a float between 0.0 and 1.0,
  "tags": a list of short strings

Example: {"is_nature": true, "confidence": 0.85, "tags": ["forest", "sky", "river"]}
Return raw JSON — no markdown fences, no extra text.
"""

# Threshold: images with nature_confidence >= this are flagged as nature.
NATURE_CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# OpenCV colour-histogram helpers (fast, supplementary data)
# ---------------------------------------------------------------------------

def _compute_colour_ratios(image_path: str) -> tuple[float, float, float]:
    """Compute green, blue (sky/water), and earth-tone pixel ratios using
    HSV colour analysis.

    Returns (green_ratio, blue_ratio, earth_ratio) each in [0.0, 1.0].
    """
    if not _HAS_CV2:
        return 0.0, 0.0, 0.0

    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        del img  # free BGR image immediately
        total_pixels = float(hsv.shape[0] * hsv.shape[1])

        # Green: H 35–85, S 40–255, V 40–255
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = float(cv2.countNonZero(green_mask)) / total_pixels
        del green_mask

        # Blue (sky/water): H 90–130, S 40–255, V 40–255
        blue_mask = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
        blue_ratio = float(cv2.countNonZero(blue_mask)) / total_pixels
        del blue_mask

        # Earth tones (brown/tan): H 10–30, S 40–200, V 40–200
        earth_mask = cv2.inRange(hsv, (10, 40, 40), (30, 200, 200))
        earth_ratio = float(cv2.countNonZero(earth_mask)) / total_pixels
        del earth_mask, hsv

        return (
            round(green_ratio, 4),
            round(blue_ratio, 4),
            round(earth_ratio, 4),
        )
    except Exception as exc:
        logger.debug("Colour analysis failed for %s: %s", image_path, exc)
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Vision-model helpers
# ---------------------------------------------------------------------------

def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_vision_response(text: str) -> tuple[bool, float, list[str]]:
    """Parse the vision model's JSON response.

    Returns (is_nature, confidence, tags).
    Falls back to heuristic parsing if JSON is invalid.
    """
    # Strip markdown fences if the model wrapped the response
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        is_nature = bool(data.get("is_nature", False))
        confidence = float(data.get("confidence", 0.0))
        tags = list(data.get("tags", []))
        return is_nature, min(max(confidence, 0.0), 1.0), tags
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.debug("Could not parse vision response as JSON: %s", text[:200])

    # Heuristic fallback: look for keywords
    lower = text.lower()
    nature_keywords = [
        "nature", "outdoor", "forest", "tree", "mountain", "sky",
        "river", "lake", "ocean", "grass", "flower", "sunset",
        "landscape", "garden", "field", "meadow", "waterfall",
    ]
    hits = [kw for kw in nature_keywords if kw in lower]
    if hits:
        return True, min(len(hits) / 5.0, 1.0), hits
    return False, 0.0, []


def _classify_nature_scene(
    image_path: str,
    llm: ChatOllama,
) -> NatureSceneResult:
    """Classify a single image as a nature scene using the vision model."""
    result = NatureSceneResult(image_path=image_path)

    # Colour histogram (always computed, fast)
    result.green_ratio, result.blue_ratio, result.earth_ratio = (
        _compute_colour_ratios(image_path)
    )

    # Vision model classification
    try:
        b64_image = _encode_image_base64(image_path)

        message = HumanMessage(
            content=[
                {"type": "text", "text": NATURE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{b64_image}",
                },
            ]
        )
        del b64_image  # free base64 data before the (slow) LLM call

        response = llm.invoke([message])
        del message  # free the message (contains image data copy)

        reply_text = response.content if hasattr(response, "content") else str(response)
        del response

        is_nature, confidence, tags = _parse_vision_response(reply_text)

        result.is_nature = is_nature
        result.nature_confidence = round(confidence, 4)
        result.scene_tags = tags

    except Exception as exc:
        logger.warning("Vision classification failed for %s: %s", image_path, exc)
        # Fall back to colour-ratio heuristic
        nature_colour_score = (
            0.5 * result.green_ratio
            + 0.3 * result.blue_ratio
            + 0.2 * result.earth_ratio
        )
        result.nature_confidence = round(min(nature_colour_score * 3.0, 1.0), 4)
        result.is_nature = result.nature_confidence >= NATURE_CONFIDENCE_THRESHOLD

    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def nature_scene_node(state: PhotoSelectorState) -> dict:
    """LangGraph node: classify images as nature scenes.

    Processes one image at a time and frees memory after each.

    Pre-filters using ``quality_results`` — only images that are sharp and
    clean are sent to the (slow) vision model.  Low-quality images get a
    default ``NatureSceneResult`` with ``is_nature=False``.

    Reads
    -----
    ``state["image_paths"]``
    ``state["quality_results"]``

    Writes
    ------
    ``state["nature_scene_results"]``  (merged via reducer)
    """
    image_paths: list[str] = state.get("image_paths", [])
    quality_results: dict[str, QualityResult] = state.get("quality_results", {})

    results: dict[str, NatureSceneResult] = {}

    # Determine which images are worth sending to the vision model
    paths_to_classify: list[str] = []
    for path in image_paths:
        qr = quality_results.get(path)
        if qr and qr.is_sharp and qr.is_clean:
            paths_to_classify.append(path)
        else:
            # Skip low-quality images — give them a default result
            results[path] = NatureSceneResult(image_path=path)

    if not paths_to_classify:
        logger.info("No images passed quality pre-filter — skipping nature classification.")
        return {"nature_scene_results": results}

    # Initialise the vision model once
    llm = ChatOllama(
        model=DEFAULT_VISION_MODEL,
        base_url=DEFAULT_BASE_URL,
        temperature=0,
    )

    total = len(paths_to_classify)
    nature_count = 0

    logger.info(
        "Classifying %d/%d images for nature scenes (skipped %d low-quality)…",
        total, len(image_paths), len(image_paths) - total,
    )

    for idx, path in enumerate(paths_to_classify, start=1):
        logger.info("[%d/%d] Nature classification: %s", idx, total, Path(path).name)
        result = _classify_nature_scene(path, llm)
        results[path] = result
        if result.is_nature:
            nature_count += 1

        # Free memory after each image (vision model keeps large tensors)
        gc.collect()

    logger.info(
        "Nature classification complete: %d/%d classified images are nature scenes.",
        nature_count, total,
    )

    gc.collect()
    return {"nature_scene_results": results}

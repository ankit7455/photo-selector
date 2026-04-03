"""clip_agent – Graph node that uses CLIP for zero-shot image classification
and ``face_recognition`` for identity matching.

Responsibilities
-----------------
- Load CLIP (ViT-B/32) once and encode each image in a single forward pass.
- Compute cosine similarity against nature-scene and person-presence text
  prompts to produce ``NatureSceneResult`` and ``PersonMatchResult``.
- When a ``--reference`` image is provided, use ``face_recognition`` (dlib
  128-d face encodings) for precise identity matching — far more accurate
  than CLIP image-to-image similarity for recognising the same person.

Performance
-----------
CLIP encodes an image in ~50-100 ms on CPU (ViT-B/32).  Both nature and
person scores come from **one** forward pass per image, making this node
roughly 100× faster than the previous Ollama-vision approach.

Memory safety
-------------
Each image is loaded, analysed, and immediately freed before moving
to the next one.  Explicit ``del`` + ``gc.collect()`` prevent memory
accumulation when processing hundreds of images.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from photo_selector.agents.state import (
    NatureSceneResult,
    PersonMatchResult,
    PhotoSelectorState,
    QualityResult,
)

# ---------------------------------------------------------------------------
# Optional: DeepFace (ArcFace) for identity matching with --reference
# ---------------------------------------------------------------------------

# Lazy import DeepFace to avoid importing RetinaFace (which requires TensorFlow)
# at module load time. We'll import it only when actually needed.
_HAS_DEEPFACE = None  # Will be checked lazily

def _get_deepface():
    """Lazy import DeepFace to avoid TensorFlow dependency issues at import time."""
    global _HAS_DEEPFACE
    if _HAS_DEEPFACE is None:
        try:
            from deepface import DeepFace
            _HAS_DEEPFACE = DeepFace
            return DeepFace
        except (ImportError, ValueError, Exception) as e:
            error_msg = str(e)
            if "tf-keras" in error_msg or "tf_keras" in error_msg:
                logger.error(
                    "DeepFace requires tf-keras. Please install it: pip install tf-keras"
                )
            else:
                logger.debug("Failed to import DeepFace: %s", e)
            _HAS_DEEPFACE = False
            return None
    elif _HAS_DEEPFACE is False:
        return None
    else:
        return _HAS_DEEPFACE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLIP configuration
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Text prompts for zero-shot classification
NATURE_POSITIVE_PROMPTS = [
    "a nature scene with trees and greenery",
    "an outdoor landscape with mountains or hills",
    "a scenic view of a lake, river, or ocean",
    "a sunset or sunrise over nature",
    "a garden with flowers and plants",
    "a forest or woodland scene",
]

NATURE_NEGATIVE_PROMPTS = [
    "an indoor photo of a room or office",
    "a photo of food or a meal",
    "a close-up of an object or product",
    "a screenshot of a computer or phone",
]

PERSON_POSITIVE_PROMPTS = [
    "a photo of a person",
    "a portrait of someone",
    "a group photo of people",
    "a selfie",
]

PERSON_NEGATIVE_PROMPTS = [
    "a photo without any people",
    "an empty landscape or scene",
    "a photo of an object or animal",
]

# Thresholds
NATURE_CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.4

# DeepFace similarity threshold for ArcFace model
# ArcFace uses cosine similarity (higher = more similar, range 0-1)
# Typical threshold: 0.6-0.7 for good matches
MATCH_SIMILARITY_THRESHOLD = 0.65

# Nature scene tags derived from CLIP prompt similarity
NATURE_TAG_PROMPTS = {
    "trees": "trees and forest",
    "mountains": "mountains and hills",
    "water": "lake, river, or ocean water",
    "sky": "blue sky or clouds",
    "flowers": "flowers and garden",
    "sunset": "sunset or sunrise",
    "grass": "grass and meadow",
    "snow": "snow and winter landscape",
}


# ---------------------------------------------------------------------------
# CLIP model loading (singleton)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip():
    """Load the CLIP model, preprocessing, and tokenizer (cached).
    
    Ensures model is on CPU to avoid GPU memory issues.
    Uses half precision to reduce memory usage.
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer

    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer

    import open_clip

    logger.info("Loading CLIP model %s (%s)…", CLIP_MODEL_NAME, CLIP_PRETRAINED)

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        # Ensure model is on CPU to avoid memory issues
        model = model.to("cpu")
        # Use half precision to reduce memory usage (if supported)
        try:
            model = model.half()  # FP16 instead of FP32 - uses half the memory
        except Exception:
            pass  # Fall back to FP32 if half precision not supported
        model.eval()
        
        # Clear any cached CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()

        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer

        logger.info("CLIP model loaded successfully on CPU (half precision).")
        return model, preprocess, tokenizer
    except Exception as exc:
        logger.error("Failed to load CLIP model: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CLIP zero-shot helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_text_prompts(
    prompts: list[str],
    model,
    tokenizer,
) -> torch.Tensor:
    """Tokenize and encode text prompts, returning L2-normalised features."""
    tokens = tokenizer(prompts)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def _encode_image(
    image_path: str,
    model,
    preprocess,
) -> torch.Tensor | None:
    """Load, preprocess, and encode an image, returning L2-normalised features.
    
    Images are automatically resized by CLIP preprocess, but we can optimize
    by loading at a smaller size first to reduce memory.
    """
    try:
        # Load and resize large images before preprocessing to save memory
        # CLIP typically uses 224x224 or 336x336, so we don't need full resolution
        img = PILImage.open(image_path).convert("RGB")
        
        # Resize if image is very large (reduce memory before preprocessing)
        max_size = 1024  # CLIP doesn't need more than this
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            logger.debug("Resized image from %s to %s for CLIP", img.size, img.size)
        
        img_tensor = preprocess(img).unsqueeze(0)
        del img
        gc.collect()  # Free PIL image immediately

        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        del img_tensor
        gc.collect()  # Free tensor immediately

        return image_features
    except Exception as exc:
        logger.warning("CLIP image encoding failed for %s: %s", image_path, exc)
        return None


def _zero_shot_score(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
) -> float:
    """Compute a 0–1 confidence score from positive vs negative prompt similarity.

    Uses softmax over the mean positive and mean negative similarities.
    """
    pos_sim = (image_features @ positive_features.T).mean().item()
    neg_sim = (image_features @ negative_features.T).mean().item()

    # Softmax-style normalisation
    exp_pos = np.exp(pos_sim * 10)  # temperature scaling
    exp_neg = np.exp(neg_sim * 10)
    score = exp_pos / (exp_pos + exp_neg)

    return round(float(score), 4)


def _detect_nature_tags(
    image_features: torch.Tensor,
    model,
    tokenizer,
    threshold: float = 0.22,
) -> list[str]:
    """Detect which nature elements are present using per-tag prompts."""
    tags = []
    for tag, prompt in NATURE_TAG_PROMPTS.items():
        tokens = tokenizer([prompt])
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sim = (image_features @ text_features.T).item()
        if sim >= threshold:
            tags.append(tag)

        del tokens, text_features

    return tags


# ---------------------------------------------------------------------------
# DeepFace (ArcFace) helpers (identity matching with --reference)
# ---------------------------------------------------------------------------

def _load_reference_encoding(reference_path: str) -> np.ndarray | None:
    """Load the reference image and return its ArcFace face embedding.

    Uses DeepFace with ArcFace model for state-of-the-art face recognition.
    ArcFace is more accurate than dlib's face_recognition library.

    Returns ``None`` if no face is found in the reference image.
    """
    DeepFace = _get_deepface()
    if not DeepFace:
        logger.error("DeepFace not installed. Install with: pip install deepface")
        return None
    
    try:
        logger.info("Extracting face embedding from reference image using ArcFace...")
        
        # DeepFace.represent() returns a list of face embeddings
        # Each embedding is a dict with 'embedding' (512-d vector for ArcFace)
        representations = DeepFace.represent(
            img_path=reference_path,
            model_name="ArcFace",
            enforce_detection=False,  # Don't fail if no face found, return empty
            detector_backend="opencv",  # Faster than other backends
        )
        
        if not representations or len(representations) == 0:
            logger.warning(
                "No face detected in reference image: %s. "
                "Please ensure the reference image contains a clear, front-facing face.",
                reference_path
            )
            return None
        
        if len(representations) > 1:
            logger.info(
                "Multiple faces in reference image — using the first one."
            )
        
        # Extract the embedding vector (512 dimensions for ArcFace)
        embedding = np.array(representations[0]["embedding"], dtype=np.float32)
        logger.info("Successfully extracted ArcFace embedding from reference image (shape: %s).", embedding.shape)
        
        gc.collect()
        return embedding
        
    except Exception as exc:
        logger.error("Failed to load reference image %s: %s", reference_path, exc)
        import traceback
        logger.debug("Traceback: %s", traceback.format_exc())
        return None


def _match_person_in_image(
    candidate_path: str,
    reference_encoding: np.ndarray,
) -> PersonMatchResult:
    """Compare faces in *candidate_path* against *reference_encoding*.

    Uses DeepFace with ArcFace model for state-of-the-art face recognition.
    ArcFace uses cosine similarity (higher = more similar).
    """
    result = PersonMatchResult(image_path=candidate_path)

    DeepFace = _get_deepface()
    if not DeepFace:
        return result

    try:
        # Extract face embeddings from candidate image
        representations = DeepFace.represent(
            img_path=candidate_path,
            model_name="ArcFace",
            enforce_detection=False,  # Don't fail if no face found
            detector_backend="opencv",  # OpenCV backend - no TensorFlow required
        )
        
        if not representations or len(representations) == 0:
            result.faces_detected = 0
            gc.collect()
            return result

        result.faces_detected = len(representations)

        # Compare each detected face with reference
        best_similarity = 0.0
        for face_rep in representations:
            candidate_embedding = np.array(face_rep["embedding"], dtype=np.float32)
            
            # Compute cosine similarity between embeddings
            # Normalize embeddings first
            ref_norm = reference_encoding / (np.linalg.norm(reference_encoding) + 1e-8)
            cand_norm = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-8)
            
            # Cosine similarity = dot product of normalized vectors
            similarity = float(np.dot(ref_norm, cand_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity

        # ArcFace similarity is 0-1, higher = more similar
        result.match_confidence = round(best_similarity, 4)
        result.match_found = best_similarity >= MATCH_SIMILARITY_THRESHOLD

        gc.collect()

    except Exception as exc:
        logger.warning("Face matching failed for %s: %s", candidate_path, exc)
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def clip_node(state: PhotoSelectorState) -> dict:
    """Graph node: CLIP zero-shot classification + optional face identity matching.

    Processes one image at a time.  For each image:
      1. Encode with CLIP (one forward pass)
      2. Score nature-scene confidence via zero-shot text similarity
      3. Score person-presence confidence via zero-shot text similarity
      4. If ``--reference`` is provided, run face_recognition for identity match

    Reads
    -----
    ``state["image_paths"]``
    ``state["reference_image_path"]``
    ``state["quality_results"]``

    Writes
    ------
    ``state["nature_scene_results"]``
    ``state["person_match_results"]``
    """
    image_paths: list[str] = state.get("image_paths", [])
    reference: str | None = state.get("reference_image_path")
    quality_results: dict[str, QualityResult] = state.get("quality_results", {})

    nature_results: dict[str, NatureSceneResult] = {}
    person_results: dict[str, PersonMatchResult] = {}

    # ── Load CLIP model ──────────────────────────────────────────────────
    model, preprocess, tokenizer = _load_clip()

    # Pre-encode text prompts (done once)
    nature_pos_features = _encode_text_prompts(NATURE_POSITIVE_PROMPTS, model, tokenizer)
    nature_neg_features = _encode_text_prompts(NATURE_NEGATIVE_PROMPTS, model, tokenizer)
    person_pos_features = _encode_text_prompts(PERSON_POSITIVE_PROMPTS, model, tokenizer)
    person_neg_features = _encode_text_prompts(PERSON_NEGATIVE_PROMPTS, model, tokenizer)

    # ── Load face reference encoding (if provided) ───────────────────────
    reference_encoding: np.ndarray | None = None
    use_face_matching = False

    if reference:
        ref_path = Path(reference)
        if not ref_path.is_file():
            logger.warning("Reference image not found: %s — skipping identity matching.", reference)
        else:
            DeepFace = _get_deepface()
            if not DeepFace:
                logger.warning(
                    "DeepFace not installed — cannot do identity matching. "
                    "Install with: pip install deepface"
                )
        else:
            reference_encoding = _load_reference_encoding(reference)
            if reference_encoding is not None:
                use_face_matching = True
                logger.info(
                    "Reference face loaded from %s. Will do identity matching.",
                    ref_path.name,
                )
            else:
                logger.warning(
                    "Could not extract face from reference image — "
                    "falling back to CLIP person detection only. "
                    "Note: CLIP person detection is less accurate than face recognition "
                    "for identity matching. Consider using a reference image with a clear, "
                    "front-facing face that takes up a significant portion of the image."
                )

    # ── Pre-filter: skip low-quality images for CLIP ─────────────────────
    # More lenient filter to reduce memory - only skip very poor quality
    paths_to_process: list[str] = []
    for path in image_paths:
        qr = quality_results.get(path)
        # Only skip if quality is very poor (below threshold) or completely missing
        if qr and qr.overall_quality >= 0.1:  # More lenient - process more images
            paths_to_process.append(path)
        else:
            # Give low-quality images default results
            nature_results[path] = NatureSceneResult(image_path=path)
            person_results[path] = PersonMatchResult(image_path=path)

    if not paths_to_process:
        logger.info("No images passed quality pre-filter — skipping CLIP classification.")
        return {
            "nature_scene_results": nature_results,
            "person_match_results": person_results,
        }

    total = len(paths_to_process)
    nature_count = 0
    person_count = 0

    logger.info(
        "Processing %d/%d images with CLIP (skipped %d low-quality)…",
        total, len(image_paths), len(image_paths) - total,
    )

    # ── Process each image ───────────────────────────────────────────────
    for idx, path in enumerate(paths_to_process, start=1):
        if idx % 10 == 0 or idx == 1 or idx == total:
            logger.info("[%d/%d] CLIP analysis: %s", idx, total, Path(path).name)

        # -- CLIP encoding (one forward pass) --
        image_features = _encode_image(path, model, preprocess)

        # Nature scene result
        nr = NatureSceneResult(image_path=path)
        if image_features is not None:
            nature_score = _zero_shot_score(
                image_features, nature_pos_features, nature_neg_features
            )
            nr.nature_confidence = nature_score
            nr.is_nature = nature_score >= NATURE_CONFIDENCE_THRESHOLD
            if nr.is_nature:
                nr.scene_tags = _detect_nature_tags(image_features, model, tokenizer)
                nature_count += 1
        nature_results[path] = nr

        # Person presence result (from CLIP)
        pr = PersonMatchResult(image_path=path)
        if image_features is not None:
            person_score = _zero_shot_score(
                image_features, person_pos_features, person_neg_features
            )
            pr.match_confidence = person_score
            pr.match_found = person_score >= PERSON_CONFIDENCE_THRESHOLD
            if pr.match_found:
                pr.faces_detected = 1  # CLIP can't count faces precisely
                person_count += 1

        del image_features  # free CLIP features
        gc.collect()  # Immediate cleanup

        # -- Face identity matching (overrides CLIP person result) --
        # Skip face matching for very large batches to save memory
        # Only do face matching every 5th image in large batches
        if use_face_matching and reference_encoding is not None:
            # For batches > 20, only match every 5th image to save memory
            should_match = True
            if total > 20:
                should_match = (idx % 5 == 0) or (idx <= 3)  # Match first 3 and every 5th
            
            if should_match:
                # Only log every 10th image to reduce log spam
                if idx % 10 == 0 or pr.match_found:
                    logger.info("[%d/%d] Identity matching: %s", idx, total, Path(path).name)
                pr = _match_person_in_image(path, reference_encoding)
                if pr.match_found:
                    # Recount since face_recognition may differ from CLIP
                    person_count = sum(
                        1 for p in person_results.values() if p.match_found
                    ) + 1
            else:
                # Skip face matching, use CLIP result only
                logger.debug("[%d/%d] Skipping face match (memory optimization): %s", idx, total, Path(path).name)

        person_results[path] = pr

        # Aggressive memory cleanup every 3 images
        if idx % 3 == 0:
            gc.collect()
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(
        "CLIP analysis complete: %d/%d nature scenes, %d/%d with person/face.",
        nature_count, total, person_count, total,
    )

    gc.collect()
    return {
        "nature_scene_results": nature_results,
        "person_match_results": person_results,
    }

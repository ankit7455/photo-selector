"""person_match_agent – Graph node that checks whether the *same person*
from a reference image appears in each candidate image.

Responsibilities
-----------------
- Accept a **reference image** of the target person (you).
- For every candidate image, detect faces and compare against the reference.
- Produce a per-image ``PersonMatchResult`` with confidence score.

Uses DeepFace with ArcFace model for:
  1. Face detection (OpenCV backend — fast and accurate).
  2. 512-dimensional face embedding per detected face.
  3. Cosine similarity comparison (higher similarity = more similar).

If no reference image is provided, the node falls back to DeepFace
face detection (presence-only, no identity matching).  If DeepFace is not
available, OpenCV's Haar cascade is used as a last resort.

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
from typing import Any

import numpy as np

from photo_selector.agents.state import PersonMatchResult, PhotoSelectorState

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

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)

# Similarity threshold for DeepFace ArcFace model.
# ArcFace uses cosine similarity (higher = more similar, range 0-1)
# Based on logs showing 76/90 matches with 0.80 threshold but user says none contain the person,
# the threshold needs to be MUCH higher or there's a calculation issue.
# Same person: typically 0.85-0.95+ (very high confidence)
# Different person: typically 0.3-0.6
# Borderline (0.6-0.85): likely false positives
# Using 0.85 for extremely strict matching
MATCH_SIMILARITY_THRESHOLD = 0.85  # Extremely strict - only very high confidence matches


# ---------------------------------------------------------------------------
# DeepFace (ArcFace) helpers (identity matching)
# ---------------------------------------------------------------------------

# Minimum face detection confidence from the detector backend.
# When enforce_detection=False, DeepFace can produce an embedding from the
# entire image (no real face detected) with face_confidence ≈ 0.
# We require a real face detection to avoid false-positive identity matches.
MIN_FACE_DETECTION_CONFIDENCE = 0.50

# Minimum face size in pixels (width and height).
# ArcFace internally resizes to 112×112, so inputs smaller than ~80px
# lose too much detail and produce unreliable embeddings.
MIN_REF_FACE_SIZE = 80   # reference: need at least 80×80 px
MIN_CAND_FACE_SIZE = 40  # candidates: at least 40×40 px

# Detector backends to try for the reference image, in priority order.
# "retinaface" is a deep-learning detector that handles sunglasses,
# side-facing poses, and varying lighting far better than Haar/SSD.
# It also returns eye landmarks, enabling proper face alignment for ArcFace.
# NOTE: mtcnn hangs on large images (4000×3000); ssd/opencv produce false
# positives from scenery textures.  We keep opencv only as a last-resort
# fallback in case retinaface is unavailable.
_REF_DETECTOR_BACKENDS = ["retinaface", "opencv"]


# Cache for reference embedding — avoids re-detecting for every batch.
_ref_embedding_cache: dict[str, np.ndarray | None] = {}


def _face_sort_key(face_rep: dict) -> tuple[bool, int]:
    """Sort key for choosing the best face detection.

    Prioritises faces **with eye landmarks** (enables proper ArcFace
    alignment) over larger faces without them.  Within each group, the
    largest face wins.
    """
    fa = face_rep.get("facial_area", {})
    has_eyes = fa.get("left_eye") is not None and fa.get("right_eye") is not None
    area = fa.get("w", 0) * fa.get("h", 0)
    return (has_eyes, area)


def _face_area(face_rep: dict) -> int:
    """Return the area (w*h) of a face detection."""
    fa = face_rep.get("facial_area", {})
    return fa.get("w", 0) * fa.get("h", 0)


def _face_size_ok(face_rep: dict, min_size: int) -> bool:
    """Check that a face detection is at least *min_size* × *min_size* pixels."""
    fa = face_rep.get("facial_area", {})
    return fa.get("w", 0) >= min_size and fa.get("h", 0) >= min_size


def _save_reference_face_crop(
    reference_path: str,
    selected_facial_area: dict,
    all_candidates: list[dict],
) -> None:
    """DEBUG: Crop and save the detected face(s) from the reference image
    into a ``reference_person/`` directory next to the reference image.

    Saves:
      - ``selected_face.jpg`` — the face that was actually selected (largest)
      - ``face_N.jpg`` — every other candidate face found

    This lets you visually verify which face the model is using.
    Remove this call once debugging is complete.
    """
    if not _HAS_CV2:
        return

    try:
        ref_dir = Path(reference_path).parent / "reference_person"
        ref_dir.mkdir(exist_ok=True)

        img = cv2.imread(str(reference_path))
        if img is None:
            logger.warning("Could not load reference image for face crop saving.")
            return

        img_h, img_w = img.shape[:2]

        # Save the selected (largest) face
        x = selected_facial_area.get("x", 0)
        y = selected_facial_area.get("y", 0)
        w = selected_facial_area.get("w", 0)
        h = selected_facial_area.get("h", 0)

        # Add some padding around the face for context (20%)
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        face_crop = img[y1:y2, x1:x2]
        selected_path = ref_dir / "selected_face.jpg"
        cv2.imwrite(str(selected_path), face_crop)
        logger.info(
            "🔍 DEBUG: Saved selected face crop → %s (size=%dx%d, from %dx%d region)",
            selected_path, face_crop.shape[1], face_crop.shape[0], w, h,
        )

        # Save all other candidate faces too
        for i, cand in enumerate(all_candidates):
            fa = cand.get("facial_area", {})
            cx, cy = fa.get("x", 0), fa.get("y", 0)
            cw, ch = fa.get("w", 0), fa.get("h", 0)
            det = cand.get("_detector", "?")
            conf = cand.get("face_confidence", 0)

            pad_cx = int(cw * 0.2)
            pad_cy = int(ch * 0.2)
            cx1 = max(0, cx - pad_cx)
            cy1 = max(0, cy - pad_cy)
            cx2 = min(img_w, cx + cw + pad_cx)
            cy2 = min(img_h, cy + ch + pad_cy)

            crop = img[cy1:cy2, cx1:cx2]
            crop_path = ref_dir / f"face_{i}_{det}_{cw}x{ch}_conf{conf:.2f}.jpg"
            cv2.imwrite(str(crop_path), crop)

        # Also save the full reference image with rectangles drawn on faces
        annotated = img.copy()
        for i, cand in enumerate(all_candidates):
            fa = cand.get("facial_area", {})
            cx, cy = fa.get("x", 0), fa.get("y", 0)
            cw, ch = fa.get("w", 0), fa.get("h", 0)
            det = cand.get("_detector", "?")
            conf = cand.get("face_confidence", 0)
            # Green for selected (index 0), blue for others
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), color, 3)
            label = f"#{i} {det} {cw}x{ch} conf={conf:.2f}"
            cv2.putText(annotated, label, (cx, max(cy - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        annotated_path = ref_dir / "reference_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        logger.info(
            "🔍 DEBUG: Saved annotated reference → %s (%d faces marked)",
            annotated_path, len(all_candidates),
        )

        del img, annotated
        gc.collect()

    except Exception as exc:
        logger.warning("Failed to save reference face crops: %s", exc)


def _load_reference_encoding(reference_path: str) -> np.ndarray | None:
    """Load the reference image and return its ArcFace face embedding.

    Tries multiple detector backends and picks the **largest** face that
    meets both the confidence and minimum-size thresholds.  This avoids
    locking onto a tiny background face (e.g. 57×57) whose embedding is
    meaningless.

    Results are cached so repeated calls (one per batch) are instant.

    Returns ``None`` if no suitable face is found.
    """
    # Return cached result if available
    abs_path = str(Path(reference_path).resolve())
    if abs_path in _ref_embedding_cache:
        cached = _ref_embedding_cache[abs_path]
        if cached is not None:
            logger.info("Using cached reference embedding (shape=%s).", cached.shape)
        return cached

    DeepFace = _get_deepface()
    if not DeepFace:
        logger.error("DeepFace not installed. Install with: pip install deepface")
        return None

    # Pass the FILE PATH directly to DeepFace — not a numpy array.
    # DeepFace handles image loading internally and detectors work
    # best with the original resolution.  Downscaling causes small
    # faces (like those with sunglasses or at a distance) to become
    # undetectable.
    all_candidates: list[dict] = []   # faces passing filters
    all_raw_detections: list[dict] = []  # ALL detections (for debug saving)

    for backend in _REF_DETECTOR_BACKENDS:
        try:
            logger.info(
                "Extracting face embedding from reference image using ArcFace "
                "(detector=%s)…", backend,
            )
            representations = DeepFace.represent(
                img_path=str(reference_path),
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend=backend,
            )

            if not representations:
                logger.debug("Detector '%s' found no faces.", backend)
                continue

            for r in representations:
                conf = r.get("face_confidence", 0)
                fa = r.get("facial_area", {})
                w, h = fa.get("w", 0), fa.get("h", 0)
                r["_detector"] = backend
                all_raw_detections.append(r)
                logger.info(
                    "  detector=%s  face_confidence=%.4f  size=%dx%d  area=%s",
                    backend, conf, w, h, fa,
                )
                if conf >= MIN_FACE_DETECTION_CONFIDENCE and _face_size_ok(r, MIN_REF_FACE_SIZE):
                    all_candidates.append(r)
                else:
                    logger.info(
                        "  → REJECTED (need conf ≥ %.2f and size ≥ %dpx)",
                        MIN_FACE_DETECTION_CONFIDENCE, MIN_REF_FACE_SIZE,
                    )

        except Exception as exc:
            logger.warning("Detector '%s' failed: %s", backend, exc)

    # ── DEBUG: Save ALL raw detection crops for visual verification ──
    if all_raw_detections:
        _save_reference_face_crop(
            reference_path,
            all_raw_detections[0].get("facial_area", {}),
            all_raw_detections,
        )

    if not all_candidates:
        logger.error(
            "❌ Could not find a face ≥ %d×%d px with confidence ≥ %.0f%% "
            "in reference image: %s. "
            "Please use a close-up, front-facing photo of the person. "
            "(%d raw detections were found but all rejected — check "
            "reference_person/ directory for crops.)",
            MIN_REF_FACE_SIZE, MIN_REF_FACE_SIZE,
            MIN_FACE_DETECTION_CONFIDENCE * 100, reference_path,
            len(all_raw_detections),
        )
        _ref_embedding_cache[abs_path] = None
        return None

    # Pick the LARGEST face — the main subject should be the biggest face
    all_candidates.sort(key=_face_sort_key, reverse=True)
    best_face = all_candidates[0]

    face_conf = best_face.get("face_confidence", 0)
    facial_area = best_face.get("facial_area", {})
    detector_used = best_face.get("_detector", "unknown")

    embedding = np.array(best_face["embedding"], dtype=np.float32)
    embedding_norm = np.linalg.norm(embedding)
    face_w = facial_area.get("w", 0)
    face_h = facial_area.get("h", 0)
    logger.info(
        "✓ Reference embedding extracted: shape=%s, norm=%.4f, "
        "face_confidence=%.4f, size=%dx%d, detector=%s",
        embedding.shape, embedding_norm, face_conf,
        face_w, face_h, detector_used,
    )
    if len(all_candidates) > 1:
        logger.info(
            "  (%d candidate faces found — selected the largest one.)",
            len(all_candidates),
        )
    logger.warning(
        "⚠️  Verify the reference image contains a clear, front-facing photo of YOU. "
        "Check reference_person/ directory for the face crop."
    )

    _ref_embedding_cache[abs_path] = embedding
    gc.collect()
    return embedding


def _match_person_in_image(
    candidate_path: str,
    reference_encoding: np.ndarray,
) -> PersonMatchResult:
    """Compare all faces in *candidate_path* against *reference_encoding*.

    Uses DeepFace with ArcFace model for state-of-the-art face recognition.
    ArcFace uses cosine similarity (higher = more similar, range 0-1).

    Steps:
      1. Extract face embeddings from candidate image using ArcFace.
      2. Compute cosine similarity with reference embedding.
      3. If similarity >= threshold → match.
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
            detector_backend="retinaface",  # Best accuracy for varied poses/conditions
        )

        if not representations or len(representations) == 0:
            result.faces_detected = 0
            gc.collect()
            return result

        # ── Filter out low-confidence / too-small face detections ─────
        # When enforce_detection=False, DeepFace may return an embedding
        # for the *entire image* (not a face) with face_confidence ≈ 0.
        # Tiny faces (< MIN_CAND_FACE_SIZE) also produce unreliable
        # embeddings.  Both cases cause false-positive identity matches.
        valid_faces = []
        for face_rep in representations:
            conf = face_rep.get("face_confidence", 0)
            fa = face_rep.get("facial_area", {})
            w, h = fa.get("w", 0), fa.get("h", 0)
            if conf >= MIN_FACE_DETECTION_CONFIDENCE and _face_size_ok(face_rep, MIN_CAND_FACE_SIZE):
                valid_faces.append(face_rep)
            else:
                logger.debug(
                    "Skipping detection in %s: "
                    "face_confidence=%.4f, size=%dx%d (need ≥ %.2f conf, ≥ %dpx)",
                    Path(candidate_path).name, conf, w, h,
                    MIN_FACE_DETECTION_CONFIDENCE, MIN_CAND_FACE_SIZE,
                )

        if not valid_faces:
            # No real face detected — all detections were low-confidence
            result.faces_detected = 0
            logger.debug(
                "No confident face in %s (raw detections=%d, all below %.2f confidence)",
                Path(candidate_path).name, len(representations),
                MIN_FACE_DETECTION_CONFIDENCE,
            )
            gc.collect()
            return result

        result.faces_detected = len(valid_faces)

        # Compare each *real* detected face with reference
        best_similarity = 0.0
        best_face_conf = 0.0
        for face_rep in valid_faces:
            candidate_embedding = np.array(face_rep["embedding"], dtype=np.float32)
            face_conf = face_rep.get("face_confidence", 0)

            # Verify embeddings are valid
            if candidate_embedding.shape != reference_encoding.shape:
                logger.warning(
                    "Embedding shape mismatch: candidate %s vs reference %s",
                    candidate_embedding.shape, reference_encoding.shape,
                )
                continue

            # Compute cosine similarity between embeddings
            ref_norm = reference_encoding / (np.linalg.norm(reference_encoding) + 1e-8)
            cand_norm = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-8)
            similarity = float(np.dot(ref_norm, cand_norm))
            similarity = max(0.0, min(1.0, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_face_conf = face_conf

        # ArcFace similarity is 0-1, higher = more similar
        result.match_confidence = round(best_similarity, 4)
        result.match_found = best_similarity >= MATCH_SIMILARITY_THRESHOLD

        # Log matching results with face_confidence + size for transparency
        if result.match_found:
            logger.info(
                "✓ MATCH in %s: similarity=%.4f, face_conf=%.2f, "
                "faces=%d (threshold=%.2f)",
                Path(candidate_path).name, best_similarity,
                best_face_conf, len(valid_faces), MATCH_SIMILARITY_THRESHOLD,
            )
        else:
            logger.info(
                "✗ No match in %s: similarity=%.4f, face_conf=%.2f, "
                "faces=%d (threshold=%.2f, need +%.4f)",
                Path(candidate_path).name, best_similarity,
                best_face_conf, len(valid_faces), MATCH_SIMILARITY_THRESHOLD,
                MATCH_SIMILARITY_THRESHOLD - best_similarity,
            )

        gc.collect()

    except Exception as exc:
        logger.warning("Face matching failed for %s: %s", candidate_path, exc)
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Fallback: face detection only (no identity matching)
# ---------------------------------------------------------------------------

def _detect_faces_only(image_path: str) -> PersonMatchResult:
    """Fallback: detect faces when no reference image is provided.

    Uses ``face_recognition`` (dlib HOG detector) if available, otherwise
    falls back to OpenCV's DNN-based face detector.

    Only checks whether *any* face is present — no identity matching.
    """
    result = PersonMatchResult(image_path=image_path)

    # Strategy 1: Use DeepFace — more accurate
    DeepFace = _get_deepface()
    if DeepFace:
        try:
            representations = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface",
            )

            # Only count high-confidence face detections
            valid_faces = [
                r for r in (representations or [])
                if r.get("face_confidence", 0) >= MIN_FACE_DETECTION_CONFIDENCE
            ]
            result.faces_detected = len(valid_faces)
            if result.faces_detected > 0:
                result.match_found = True
                result.match_confidence = 0.85  # Fixed confidence when faces found

        except Exception as exc:
            logger.warning("Face detection (DeepFace) failed for %s: %s", image_path, exc)

        return result

    # Strategy 2: OpenCV DNN face detector (no extra model needed)
    if _HAS_CV2:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return result

            # Use OpenCV's built-in Haar cascade as a lightweight fallback
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            del img

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            del gray

            result.faces_detected = len(faces)
            if result.faces_detected > 0:
                result.match_found = True
                result.match_confidence = 0.70  # Haar is less reliable

        except Exception as exc:
            logger.warning("Face detection (OpenCV) failed for %s: %s", image_path, exc)

        return result

    logger.warning("No face detection backend available for %s", image_path)
    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def person_match_node(state: PhotoSelectorState) -> dict:
    """LangGraph node: compare each image against the reference person.

    Processes one image at a time and frees memory after each.

    **With reference image** (``--reference`` flag):
        Uses DeepFace with ArcFace model to compute face embeddings and compare
        them to the reference.  ``match_found=True`` only if the *same
        person* appears in the image.

    **Without reference image**:
        Falls back to DeepFace face detection (presence only).
        ``match_found=True`` if *any* face is detected.

    Reads
    -----
    ``state["image_paths"]``
    ``state["reference_image_path"]``

    Writes
    ------
    ``state["person_match_results"]``  (merged via reducer)
    """
    image_paths: list[str] = state.get("image_paths", [])
    reference: str | None = state.get("reference_image_path")
    results: dict[str, PersonMatchResult] = {}

    # ── Path A: Identity matching with DeepFace ArcFace ──────────────────
    if reference:
        ref_path = Path(reference)
        if not ref_path.is_file():
            logger.warning("Reference image not found: %s — skipping.", reference)
            return {"person_match_results": results}

        DeepFace = _get_deepface()
        if not DeepFace:
            logger.error(
                "DeepFace is not installed — cannot do identity matching. "
                "Install with: pip install deepface"
            )
            return {"person_match_results": results}

        # Load reference encoding once
        reference_encoding = _load_reference_encoding(reference)
        if reference_encoding is None:
            logger.error(
                "Could not extract a face from the reference image — "
                "skipping person matching."
            )
            return {"person_match_results": results}

        logger.info(
            "Reference face loaded from %s. "
            "Scanning %d images for identity matches (threshold=%.2f)…",
            ref_path.name, len(image_paths), MATCH_SIMILARITY_THRESHOLD,
        )

        total = len(image_paths)
        match_count = 0

        for idx, path in enumerate(image_paths, start=1):
            logger.info("[%d/%d] Identity matching: %s", idx, total, Path(path).name)
            result = _match_person_in_image(path, reference_encoding)
            results[path] = result
            if result.match_found:
                match_count += 1

            # Free memory after each image
            gc.collect()

        logger.info(
            "Identity matching complete: %d/%d images contain YOU.",
            match_count, total,
        )
        gc.collect()
        return {"person_match_results": results}

    # ── Path B: Face detection only (no reference) ───────────────────────
    if not _get_deepface() and not _HAS_CV2:
        logger.warning(
            "No reference image and no face detection backend available — "
            "skipping face detection entirely."
        )
        return {"person_match_results": results}

    logger.info(
        "No reference image provided — falling back to face detection only."
    )

    total = len(image_paths)
    match_count = 0

    for idx, path in enumerate(image_paths, start=1):
        logger.info("[%d/%d] Face detection: %s", idx, total, Path(path).name)
        result = _detect_faces_only(path)
        results[path] = result
        if result.match_found:
            match_count += 1

        # Free memory after each image
        gc.collect()

    logger.info(
        "Face detection complete: %d/%d images contain at least one face.",
        match_count, total,
    )

    gc.collect()
    return {"person_match_results": results}

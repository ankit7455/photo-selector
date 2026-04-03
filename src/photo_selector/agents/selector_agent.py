"""selector_agent – Final graph node that combines quality, person-match,
and nature-scene results using a deterministic scoring function to rank
and select the best images.

Responsibilities
-----------------
- Collect ``quality_results``, ``person_match_results``, and
  ``nature_scene_results`` from prior nodes.
- Compute a weighted composite score per image.
- Rank images by score and select the top N.
- Filter out duplicate/similar images to ensure diversity.
- Write the final ``selected_images`` list into the state.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from photo_selector.agents.state import (
    ImageAnalysis,
    NatureSceneResult,
    PersonMatchResult,
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
# Scoring configuration
# ---------------------------------------------------------------------------

# Weights for the composite score (must sum to 1.0).
# Note: Nature scene detection removed (CLIP model removed to save memory)
QUALITY_WEIGHT = 0.70       # sharpness, resolution, no artifacts, no corner issues
PERSON_MATCH_WEIGHT = 0.30  # face presence (DeepFace ArcFace)
NATURE_SCENE_WEIGHT = 0.00  # disabled - CLIP removed

# Images below this quality threshold are always rejected.
MIN_QUALITY_THRESHOLD = 0.15  # Balanced threshold for quality photos

# Default: select the top N images.  0 = select all above threshold.
DEFAULT_TOP_N = 100

# Similarity threshold for duplicate detection (0.0 - 1.0)
# Images with similarity above this threshold are considered duplicates
# Lower values = stricter (fewer duplicates allowed)
SIMILARITY_THRESHOLD = 0.85  # 85% similarity = duplicate


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _compute_score(
    quality: QualityResult | None,
    person_match: PersonMatchResult | None,
    nature_scene: NatureSceneResult | None,
    quality_weight: float | None = None,
    person_match_weight: float | None = None,
    nature_scene_weight: float | None = None,
) -> tuple[float, str]:
    """Compute a composite score for one image and return (score, reason).

    Scoring breakdown (default weights):
        quality       × 0.70  (overall_quality: resolution + sharpness + noise + artifacts + corner issues + framing issues)
        person_match  × 0.30  (match_confidence: DeepFace ArcFace similarity score, gradient 0-1)
        nature_scene  × 0.00  (disabled - CLIP model removed to save memory)
    
    When reference is provided, weights are adjusted to:
        quality       × 0.50
        person_match  × 0.50  (higher weight to prioritize person match)

    An image with no quality data or below the minimum quality threshold
    gets a score of 0.0.
    
    Corner and framing issues are already penalized in the overall_quality calculation,
    so no additional penalty is applied here to avoid double-penalizing.
    """
    # Use provided weights or defaults
    q_weight = quality_weight if quality_weight is not None else QUALITY_WEIGHT
    p_weight = person_match_weight if person_match_weight is not None else PERSON_MATCH_WEIGHT
    n_weight = nature_scene_weight if nature_scene_weight is not None else NATURE_SCENE_WEIGHT
    if quality is None:
        return 0.0, "No quality data available."

    q_score = quality.overall_quality

    # Note: Corner and framing issues are already penalized in overall_quality calculation
    # No additional penalty needed here to avoid double-penalizing

    if q_score < MIN_QUALITY_THRESHOLD:
        return 0.0, (
            f"Quality {q_score:.0%} is below minimum threshold "
            f"({MIN_QUALITY_THRESHOLD:.0%})."
        )

    # --- Person match component ---
    # Use similarity score (match_confidence) as gradient, but only for reasonable matches
    # If similarity is too low (< 0.80), treat as "no match" and give 0.0
    # This ensures images with clearly different people don't get person match points
    p_score = 0.0
    if person_match is not None:
        similarity = person_match.match_confidence
        # Only give person match points if similarity is reasonably high (>= 0.80)
        # Below 0.80, it's clearly not the same person, so treat as no match
        if similarity >= 0.80:
            p_score = similarity  # Use gradient: 0.80-1.0
        else:
            p_score = 0.0  # Below 0.80 = no match, only quality matters

    # --- Nature scene component (disabled - CLIP removed) ---
    n_score = 0.0
    # if nature_scene is not None:
    #     n_score = nature_scene.nature_confidence

    # --- Weighted composite ---
    composite = (
        q_weight * q_score
        + p_weight * p_score
        + n_weight * n_score
    )

    # --- Build human-readable reason ---
    parts = [
        f"quality={q_score:.0%} (×{q_weight})",
        f"person={p_score:.0%} (×{p_weight})",
    ]
    # Nature scene removed (CLIP model removed to save memory)

    flags: list[str] = []
    if person_match and person_match.match_found:
        flags.append("face detected")
    # Nature scene flag removed (CLIP model removed)
    if quality.is_sharp:
        flags.append("sharp")
    if quality.has_artifacts:
        flags.append("has artifacts")
    if quality.has_corner_issues:
        flags.append("corner issues")
    if quality.has_framing_issues:
        flags.append("framing issues")

    reason = f"Score {composite:.0%}: {', '.join(parts)}"
    if flags:
        reason += f" [{'; '.join(flags)}]"

    return round(composite, 4), reason


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _compute_image_hash(image_path: str) -> np.ndarray | None:
    """Compute a perceptual hash for an image using average hash algorithm.
    
    Returns a normalized 8x8 hash array, or None if image cannot be loaded.
    This hash is robust to minor variations (brightness, contrast) but will
    catch near-duplicate images.
    
    Loads images at reduced size to save memory.
    """
    if not _HAS_CV2:
        return None
    
    try:
        # Load at reduced size - hash only needs small image
        img = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_2)  # Load at 1/2 size
        if img is None:
            # Fallback to normal load if reduced fails
            img = cv2.imread(str(image_path))
            if img is None:
                return None
        
        # Resize to 8x8 for hash computation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Compute average
        avg = np.mean(resized)
        
        # Create hash: 1 if pixel > average, 0 otherwise
        hash_array = (resized > avg).astype(np.uint8)
        
        del img, gray, resized
        gc.collect()
        return hash_array
        
    except Exception as exc:
        logger.debug("Failed to compute hash for %s: %s", image_path, exc)
        return None


def _compute_histogram(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute color histogram for an image.
    
    Returns a tuple of (hist_b, hist_g, hist_r) normalized histograms,
    or None if image cannot be loaded.
    
    Loads images at reduced size to save memory.
    """
    if not _HAS_CV2:
        return None
    
    try:
        # Load at reduced size - histogram doesn't need full resolution
        img = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)  # Load at 1/4 size
        if img is None:
            # Fallback to normal load if reduced fails
            img = cv2.imread(str(image_path))
            if img is None:
                return None
        
        # Compute histogram for each channel (BGR)
        hist_b = cv2.calcHist([img], [0], None, [50], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [50], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [50], [0, 256])
        
        # Normalize each histogram
        hist_b = cv2.normalize(hist_b, hist_b)
        hist_g = cv2.normalize(hist_g, hist_g)
        hist_r = cv2.normalize(hist_r, hist_r)
        
        del img
        gc.collect()
        return (hist_b, hist_g, hist_r)
        
    except Exception as exc:
        logger.debug("Failed to compute histogram for %s: %s", image_path, exc)
        return None


def _images_are_similar(
    path1: str,
    path2: str,
    hash1: np.ndarray | None = None,
    hash2: np.ndarray | None = None,
    hist1: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    hist2: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Check if two images are similar (near-duplicates).
    
    Uses a combination of perceptual hashing and histogram comparison.
    Returns True if images are considered duplicates.
    """
    # Method 1: Perceptual hash comparison (fastest, most accurate for near-duplicates)
    if hash1 is not None and hash2 is not None:
        # Hamming distance between hashes
        hamming_distance = np.sum(hash1 != hash2)
        # 8x8 = 64 bits, so max distance is 64
        # Similarity = 1 - (distance / 64)
        hash_similarity = 1.0 - (hamming_distance / 64.0)
        
        if hash_similarity >= SIMILARITY_THRESHOLD:
            return True
    
    # Method 2: Histogram comparison (fallback, good for color similarity)
    if hist1 is not None and hist2 is not None:
        # Compare each channel and average the similarity
        correlations = []
        for h1, h2 in zip(hist1, hist2):
            corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        if avg_correlation >= SIMILARITY_THRESHOLD:
            return True
    
    # If we can't compute similarity, assume they're different
    return False


def _filter_duplicates(
    scored_entries: list[tuple[str, float, str]],
    max_images: int,
) -> list[tuple[str, float, str]]:
    """Filter out duplicate/similar images from the scored list.
    
    Iterates through images in score order and skips images that are too
    similar to already selected images.
    
    Args:
        scored_entries: List of (path, score, reason) tuples, sorted by score (highest first)
        max_images: Maximum number of images to select
    
    Returns:
        Filtered list of unique images
    """
    if not _HAS_CV2:
        logger.warning("OpenCV not available — skipping duplicate detection")
        return scored_entries[:max_images] if max_images > 0 else scored_entries
    
    selected: list[tuple[str, float, str]] = []
    selected_hashes: dict[str, np.ndarray] = {}  # path -> hash
    selected_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # path -> (hist_b, hist_g, hist_r)
    
    logger.info("Filtering duplicates from %d scored images...", len(scored_entries))
    
    for path, score, reason in scored_entries:
        if max_images > 0 and len(selected) >= max_images:
            break
        
        # Compute hash and histogram for current image
        current_hash = _compute_image_hash(path)
        current_hist = _compute_histogram(path)
        
        # Check similarity against already selected images
        # Only check hash first (faster), then histogram if hash is similar
        is_duplicate = False
        for selected_path, _, _ in selected:
            selected_hash = selected_hashes.get(selected_path)
            
            # Quick hash check first (much faster)
            if current_hash is not None and selected_hash is not None:
                hamming_distance = np.sum(current_hash != selected_hash)
                hash_similarity = 1.0 - (hamming_distance / 64.0)
                if hash_similarity >= SIMILARITY_THRESHOLD:
                    # Hash match - check histogram for confirmation
                    selected_hist = selected_hists.get(selected_path)
                    if current_hist is not None and selected_hist is not None:
                        if _images_are_similar(
                            path,
                            selected_path,
                            hash1=current_hash,
                            hash2=selected_hash,
                            hist1=current_hist,
                            hist2=selected_hist,
                        ):
                            is_duplicate = True
                            logger.debug(
                                "Skipping duplicate: %s (similar to %s)",
                                Path(path).name,
                                Path(selected_path).name,
                            )
                            break
                    else:
                        # Hash match but no histograms - assume duplicate
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            selected.append((path, score, reason))
            if current_hash is not None:
                selected_hashes[path] = current_hash
            if current_hist is not None:
                selected_hists[path] = current_hist
        
        # Free memory periodically (less frequently)
        if len(selected) % 20 == 0:
            gc.collect()
    
    duplicates_removed = len(scored_entries) - len(selected)
    if duplicates_removed > 0:
        logger.info(
            "Duplicate filtering complete: %d duplicates removed, %d unique images selected",
            duplicates_removed,
            len(selected),
        )
    else:
        logger.info("No duplicates found among top images")
    
    gc.collect()
    return selected


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def selector_node(state: PhotoSelectorState) -> dict:
    """LangGraph node: deterministically score, rank, and select the best images.

    Reads
    -----
    ``state["image_paths"]``
    ``state["quality_results"]``
    ``state["person_match_results"]``
    ``state["nature_scene_results"]``

    Writes
    ------
    ``state["analyses"]``
    ``state["selected_images"]``
    """
    image_paths: list[str] = state.get("image_paths", [])
    quality: dict[str, QualityResult] = state.get("quality_results", {})
    person: dict[str, PersonMatchResult] = state.get("person_match_results", {})
    nature: dict[str, NatureSceneResult] = state.get("nature_scene_results", {})
    top_n: int = state.get("top_n", DEFAULT_TOP_N)
    reference_provided: str | None = state.get("reference_image_path")
    
    # Adjust weights when reference is provided - give more weight to person match
    if reference_provided:
        # When reference provided: prioritize person match more (50/50 split)
        quality_weight = 0.50
        person_match_weight = 0.50
        nature_scene_weight = 0.00
        logger.info(
            "Reference provided: person match weight increased to 50% (from 30%). "
            "All images considered, but those with higher similarity scores will rank higher."
        )
    else:
        # When no reference: use default weights
        quality_weight = QUALITY_WEIGHT
        person_match_weight = PERSON_MATCH_WEIGHT
        nature_scene_weight = NATURE_SCENE_WEIGHT

    # -- Score every image ------------------------------------------------
    scored: list[tuple[str, float, str]] = []  # (path, score, reason)

    for path in image_paths:
        q = quality.get(path)
        p = person.get(path)
        n = nature.get(path)
        # Pass weights to scoring function
        score, reason = _compute_score(q, p, n, quality_weight, person_match_weight, nature_scene_weight)
        scored.append((path, score, reason))
        logger.debug("Scored %s → %.4f", path, score)

    # -- Rank by score (highest first) ------------------------------------
    scored.sort(key=lambda x: x[1], reverse=True)

    # -- Filter out duplicates first, then select top N -------------------
    # Filter duplicates from all scored images (only consider images with score > 0)
    candidates = [(p, s, r) for p, s, r in scored if s > 0]
    
    # Remove duplicates from all candidates (processes in score order)
    unique_entries = _filter_duplicates(candidates, len(candidates))
    
    # Now select top N from the deduplicated list
    if top_n > 0:
        selected_entries = unique_entries[:top_n]
    else:
        selected_entries = unique_entries

    selected_paths = [p for p, _, _ in selected_entries]
    selected_set = set(selected_paths)

    # -- Build per-image analyses -----------------------------------------
    analyses: dict[str, ImageAnalysis] = {}
    rank_counter = 0

    for path, score, reason in scored:
        is_selected = path in selected_set
        if is_selected:
            rank_counter += 1
            rank = rank_counter
        else:
            rank = 0

        analyses[path] = ImageAnalysis(
            image_path=path,
            quality=quality.get(path),
            person_match=person.get(path),
            nature_scene=nature.get(path),
            selected=is_selected,
            rank=rank,
            composite_score=score,
            reason=reason,
        )

    logger.info(
        "Selection complete: %d/%d images selected (top_n=%d).",
        len(selected_paths), len(image_paths), top_n,
    )

    return {
        "analyses": analyses,
        "selected_images": selected_paths,
    }

"""quality_agent – Graph node that evaluates image quality.

Responsibilities
-----------------
- Resolution (width × height, megapixels)
- Blurriness detection (Laplacian variance via OpenCV)
- Noise estimation (standard deviation in high-frequency domain)
- Pixel artifact / breakage detection (block-boundary analysis)
- Corner/edge issue detection (unwanted objects in photo corners and edges)
- Framing/composition analysis (detects poor framing, edge cutoffs, unbalanced composition)
- Produce a normalised 0–1 ``overall_quality`` score optimized for Instagram-ready photos

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

from photo_selector.agents.state import PhotoSelectorState, QualityResult

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image as _PILImage

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality analysis helpers
# ---------------------------------------------------------------------------

def _compute_blur_score(gray: Any) -> float:
    """Laplacian variance — higher values mean sharper images.

    The Laplacian operator highlights rapid intensity changes (edges).
    A blurry image will have very few edges, resulting in low variance.
    Typical thresholds:
        < 50   → likely blurry
        50–100 → borderline
        > 100  → acceptably sharp
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = float(laplacian.var())
    del laplacian
    return score


def _compute_noise_score(gray: Any) -> float:
    """Estimate noise level using the median absolute deviation of the
    high-frequency component (wavelet-like approach using a 3×3 kernel).

    Returns a normalised 0–1 score where lower is better (less noise).
    """
    # High-pass filter: original − blurred  →  noise residual
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float64) - blurred.astype(np.float64)
    sigma = np.std(residual)
    del blurred, residual

    # Normalise: sigma of ~30+ is very noisy → 1.0
    return float(min(sigma / 30.0, 1.0))


def _detect_artifacts(gray: Any, block_size: int = 8) -> bool:
    """Detect JPEG/compression artifacts by analysing block-boundary
    discontinuities.

    JPEG compresses in 8×8 blocks; artifact-heavy images show strong
    edges exactly at block boundaries.  We compare the average gradient
    *at* block boundaries vs. the average gradient *within* blocks.

    Returns ``True`` if artefacts are likely.
    """
    h, w = gray.shape[:2]
    if h < block_size * 4 or w < block_size * 4:
        return False  # image too small to analyse

    # Compute absolute horizontal gradient
    grad_x = np.abs(np.diff(gray.astype(np.float64), axis=1))

    # Indices of block boundaries (columns that are multiples of block_size)
    boundary_cols = list(range(block_size - 1, grad_x.shape[1], block_size))
    if not boundary_cols:
        del grad_x
        return False

    boundary_mean = float(np.mean(grad_x[:, boundary_cols]))
    # Non-boundary columns
    all_cols = set(range(grad_x.shape[1]))
    non_boundary_cols = sorted(all_cols - set(boundary_cols))
    if not non_boundary_cols:
        del grad_x
        return False

    non_boundary_mean = float(np.mean(grad_x[:, non_boundary_cols]))
    del grad_x

    # If boundary gradient is significantly higher → artefacts present
    if non_boundary_mean == 0:
        return boundary_mean > 5.0
    ratio = boundary_mean / non_boundary_mean
    return ratio > 1.5


def _detect_corner_issues(img_cv: Any, edge_region_ratio: float = 0.12) -> bool:
    """Detect unwanted objects or issues in photo corners and edges.
    
    This function analyzes corner regions AND edge regions (top, bottom, left, right)
    to detect if there are significant objects or anomalies that might make
    the photo imperfect. Stricter detection for Instagram-ready photos.
    
    Args:
        img_cv: OpenCV BGR image
        edge_region_ratio: Fraction of image dimensions to consider as edge region (default 0.12)
    
    Returns:
        True if corner/edge issues are detected (unwanted objects in corners/edges)
    """
    h, w = img_cv.shape[:2]
    if h < 100 or w < 100:
        return False  # image too small to analyse
    
    # Define edge region sizes (stricter - smaller regions)
    edge_w = int(w * edge_region_ratio)
    edge_h = int(h * edge_region_ratio)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Define corner regions: top-left, top-right, bottom-left, bottom-right
    corners = [
        gray[0:edge_h, 0:edge_w],  # top-left
        gray[0:edge_h, w-edge_w:w],  # top-right
        gray[h-edge_h:h, 0:edge_w],  # bottom-left
        gray[h-edge_h:h, w-edge_w:w],  # bottom-right
    ]
    
    # Define edge regions: top, bottom, left, right (excluding corners)
    edges = [
        gray[0:edge_h, edge_w:w-edge_w],  # top edge
        gray[h-edge_h:h, edge_w:w-edge_w],  # bottom edge
        gray[edge_h:h-edge_h, 0:edge_w],  # left edge
        gray[edge_h:h-edge_h, w-edge_w:w],  # right edge
    ]
    
    # Define center region for comparison (middle 50% of image)
    center_w_start = int(w * 0.25)
    center_w_end = int(w * 0.75)
    center_h_start = int(h * 0.25)
    center_h_end = int(h * 0.75)
    center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
    
    if center_region.size == 0:
        del gray
        return False
    
    # Compute statistics for center region
    center_mean = float(np.mean(center_region))
    center_std = float(np.std(center_region))
    
    # Analyze each corner for anomalies (stricter thresholds)
    corner_issues = []
    for i, corner in enumerate(corners):
        if corner.size == 0:
            continue
        
        corner_mean = float(np.mean(corner))
        corner_std = float(np.std(corner))
        
        # Compute edge density (high edge density suggests objects)
        laplacian = cv2.Laplacian(corner, cv2.CV_64F)
        edge_density = float(np.mean(np.abs(laplacian)))
        del laplacian
        
        brightness_diff = abs(corner_mean - center_mean) / (center_mean + 1.0)
        variance_ratio = corner_std / (center_std + 1.0) if center_std > 0 else 0
        
        # Balanced thresholds - only flag obvious issues, not natural content
        # Top corners are important but shouldn't flag sky/trees
        is_top_corner = i in [0, 1]
        top_corner_threshold = 60.0 if is_top_corner else 70.0  # Higher threshold to avoid false positives
        
        has_issue = (
            edge_density > top_corner_threshold or  # Very high edge density (obvious objects)
            (brightness_diff > 0.4 and variance_ratio > 1.5) or  # Significant brightness + variance difference
            (edge_density > 45.0 and brightness_diff > 0.3)  # High edge + significant brightness diff
        )
        
        if has_issue:
            corner_issues.append(i)
    
    # Analyze edges (especially top edge for framing issues)
    edge_issues = []
    for i, edge in enumerate(edges):
        if edge.size == 0:
            continue
        
        edge_mean = float(np.mean(edge))
        edge_std = float(np.std(edge))
        
        laplacian = cv2.Laplacian(edge, cv2.CV_64F)
        edge_density = float(np.mean(np.abs(laplacian)))
        del laplacian
        
        brightness_diff = abs(edge_mean - center_mean) / (center_mean + 1.0)
        variance_ratio = edge_std / (center_std + 1.0) if center_std > 0 else 0
        
        # Top edge is important but shouldn't flag natural sky/horizon
        is_top_edge = i == 0
        edge_threshold = 55.0 if is_top_edge else 65.0  # Higher threshold
        
        has_issue = (
            edge_density > edge_threshold or  # Very high edge density
            (brightness_diff > 0.4 and variance_ratio > 1.4) or  # Significant difference
            (edge_density > 45.0 and brightness_diff > 0.3)  # High edge + significant diff
        )
        
        if has_issue:
            edge_issues.append(i)
    
    del gray
    
    # Balanced detection: Only flag if multiple issues or severe single issue
    # Don't flag single top corner/edge issues as they might be natural (sky, horizon)
    # Only flag if:
    # - 3+ corners have issues (severe problem)
    # - 2+ corners AND 1+ edge have issues
    # - Bottom-right corner (most common real issue) + another issue
    
    has_bottom_right_issue = 3 in corner_issues  # Bottom-right corner
    
    if len(corner_issues) >= 3:
        return True  # Multiple corners have issues
    
    if has_bottom_right_issue and (len(corner_issues) >= 2 or len(edge_issues) >= 1):
        return True  # Bottom-right + other issues
    
    if len(corner_issues) >= 2 and len(edge_issues) >= 2:
        return True  # Multiple corners and edges
    
    return False


def _detect_framing_issues(img_cv: Any) -> bool:
    """Detect poor framing/composition issues in photos.
    
    Analyzes:
    - Subject positioning (rule of thirds)
    - Edge cutoffs (subjects too close to edges)
    - Unbalanced composition
    - Excessive empty space at edges
    
    Returns:
        True if framing issues are detected
    """
    h, w = img_cv.shape[:2]
    if h < 200 or w < 200:
        return False  # image too small to analyse
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect edges to find subject boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Find regions with significant content (non-empty areas)
    # Use edge density to identify where subjects/content are
    kernel = np.ones((20, 20), np.uint8)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel / (20 * 20))
    
    # Define edge zones (outer 15% of image)
    edge_zone_ratio = 0.15
    edge_w = int(w * edge_zone_ratio)
    edge_h = int(h * edge_zone_ratio)
    
    # Check if significant content is too close to edges
    # Top edge
    top_zone = edge_density[0:edge_h, :]
    top_content = np.mean(top_zone) > 15.0  # High edge density in top
    
    # Bottom edge
    bottom_zone = edge_density[h-edge_h:h, :]
    bottom_content = np.mean(bottom_zone) > 15.0
    
    # Left edge
    left_zone = edge_density[:, 0:edge_w]
    left_content = np.mean(left_zone) > 15.0
    
    # Right edge
    right_zone = edge_density[:, w-edge_w:w]
    right_content = np.mean(right_zone) > 15.0
    
    # Center region should have good content
    center_w_start = int(w * 0.2)
    center_w_end = int(w * 0.8)
    center_h_start = int(h * 0.2)
    center_h_end = int(h * 0.8)
    center_zone = edge_density[center_h_start:center_h_end, center_w_start:center_w_end]
    center_content = np.mean(center_zone)
    
    del gray, edges, edge_density
    
    # Framing issues: Only flag severe problems
    # Don't flag normal compositions where subjects are naturally positioned
    edge_content_count = sum([top_content, bottom_content, left_content, right_content])
    
    # Only flag severe framing issues:
    # 1. Content at ALL 4 edges with low center (very poor composition)
    # 2. Very low center content with high edge content (subject cut off)
    has_framing_issue = (
        (edge_content_count >= 4 and center_content < 12.0) or  # All edges + low center
        (center_content < 8.0 and edge_content_count >= 3)  # Very low center + 3 edges
    )
    
    return has_framing_issue


def _evaluate_image_quality(image_path: str) -> QualityResult:
    """Return quality metrics for a single image using OpenCV + Pillow."""
    result = QualityResult(image_path=image_path)

    path = Path(image_path)
    if not path.is_file():
        logger.warning("Image file not found: %s", image_path)
        return result

    # --- Resolution via Pillow (handles HEIC/HEIF too) ---
    if not _HAS_PIL:
        logger.debug("Pillow not installed — skipping resolution analysis for %s", path)
        return result

    try:
        with _PILImage.open(path) as img:
            result.width, result.height = img.size
            result.megapixels = round(
                (result.width * result.height) / 1_000_000, 2
            )
    except Exception as exc:
        logger.warning("Pillow could not open %s: %s", path, exc)
        return result

    # --- OpenCV analysis ---
    if not _HAS_CV2:
        logger.debug("OpenCV not installed — using resolution-only quality for %s", path)
        result.overall_quality = min(result.megapixels / 12.0, 1.0)
        return result

    try:
        # Load image and resize if too large to reduce memory usage
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            logger.warning("OpenCV could not read: %s", path)
            # Fall back to resolution-only quality
            result.overall_quality = min(result.megapixels / 12.0, 1.0)
            return result

        # Resize large images to max 2048px on longest side to save memory
        # Quality analysis doesn't need full resolution
        h, w = img_cv.shape[:2]
        max_dimension = 2048
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug("Resized image from %dx%d to %dx%d for memory efficiency", w, h, new_w, new_h)

        # Corner/edge issue detection (needs full color image)
        result.has_corner_issues = _detect_corner_issues(img_cv)
        
        # Framing/composition issue detection
        result.has_framing_issues = _detect_framing_issues(img_cv)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        del img_cv  # free the full-colour image immediately
        gc.collect()  # Immediate cleanup

        # Blur score (Laplacian variance)
        result.blur_score = _compute_blur_score(gray)

        # Noise score
        result.noise_score = _compute_noise_score(gray)

        # Artifact detection
        result.has_artifacts = _detect_artifacts(gray)

        del gray  # free grayscale image
        gc.collect()  # Immediate cleanup

    except Exception as exc:
        logger.warning("OpenCV analysis failed for %s: %s", path, exc)
        result.overall_quality = min(result.megapixels / 12.0, 1.0)
        return result

    # --- Composite quality score (0.0 – 1.0) ---
    # Weight components for balanced photo quality:
    #   resolution     : 25%
    #   sharpness      : 35%  (most important for "no blurriness")
    #   noise          : 15%
    #   artifacts      : 10%
    #   corner_issues  : 10%  (penalize unwanted objects in corners/edges)
    #   framing_issues : 5%   (penalize severe framing issues)

    resolution_score = min(result.megapixels / 12.0, 1.0)

    # Normalise blur_score: typical range 0–2000+
    # Map: 0→0, 100→0.5, 500→0.9, 1000+→1.0
    sharpness_score = min(result.blur_score / 1000.0, 1.0)

    noise_quality = 1.0 - result.noise_score  # less noise = higher quality
    artifact_quality = 0.0 if result.has_artifacts else 1.0
    corner_quality = 0.0 if result.has_corner_issues else 1.0  # no corner issues = higher quality
    framing_quality = 0.0 if result.has_framing_issues else 1.0  # no framing issues = higher quality

    result.overall_quality = round(
        0.25 * resolution_score
        + 0.35 * sharpness_score
        + 0.15 * noise_quality
        + 0.10 * artifact_quality
        + 0.10 * corner_quality
        + 0.05 * framing_quality,
        4,
    )

    return result


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def quality_node(state: PhotoSelectorState) -> dict:
    """LangGraph node: analyse every image for quality metrics.

    Processes one image at a time and frees memory after each.

    Reads
    -----
    ``state["image_paths"]``

    Writes
    ------
    ``state["quality_results"]``  (merged via reducer)
    """
    image_paths: list[str] = state.get("image_paths", [])
    results: dict[str, QualityResult] = {}

    total = len(image_paths)
    logger.info("Starting quality analysis for %d images...", total)
    for idx, path in enumerate(image_paths, start=1):
        if idx % 10 == 0 or idx == 1 or idx == total:
            logger.info("[%d/%d] Analysing quality: %s", idx, total, Path(path).name)
        results[path] = _evaluate_image_quality(path)

        # Aggressive memory cleanup every 5 images
        if idx % 5 == 0:
            gc.collect()

    # Log summary
    sharp_count = sum(1 for r in results.values() if r.is_sharp)
    clean_count = sum(1 for r in results.values() if r.is_clean)
    logger.info(
        "Quality analysis complete: %d/%d sharp, %d/%d artifact-free",
        sharp_count, total, clean_count, total,
    )

    gc.collect()
    return {"quality_results": results}

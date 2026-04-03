"""state – Shared state schema that flows through the LangGraph pipeline.

Every node in the graph reads from and writes to this state.  LangGraph
merges the returned dict back into the running state automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Image-level result containers
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Quality metrics for a single image."""

    image_path: str
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    blur_score: float = 0.0       # higher = sharper (Laplacian variance)
    noise_score: float = 0.0      # 0.0 – 1.0 (lower = less noise)
    has_artifacts: bool = False    # True if pixel breakage / compression artifacts detected
    has_corner_issues: bool = False  # True if unwanted objects detected in corners/edges
    has_framing_issues: bool = False  # True if photo has poor framing/composition
    overall_quality: float = 0.0  # 0.0 – 1.0 normalised score

    @property
    def resolution_label(self) -> str:
        if self.megapixels >= 12:
            return "excellent"
        if self.megapixels >= 5:
            return "good"
        if self.megapixels >= 2:
            return "acceptable"
        return "low"

    @property
    def is_sharp(self) -> bool:
        """True if the image passes the blur threshold."""
        return self.blur_score >= 100.0  # Laplacian variance threshold

    @property
    def is_clean(self) -> bool:
        """True if the image has no significant artifacts or noise."""
        return not self.has_artifacts and self.noise_score < 0.4


@dataclass
class PersonMatchResult:
    """Person-matching result for a single image."""

    image_path: str
    faces_detected: int = 0
    match_found: bool = False
    match_confidence: float = 0.0  # 0.0 – 1.0


@dataclass
class NatureSceneResult:
    """Nature-scene classification for a single image."""

    image_path: str
    is_nature: bool = False
    nature_confidence: float = 0.0  # 0.0 – 1.0
    scene_tags: list[str] = field(default_factory=list)  # e.g. ["forest", "sky", "water"]
    green_ratio: float = 0.0   # fraction of green-dominant pixels
    blue_ratio: float = 0.0    # fraction of blue-dominant pixels (sky / water)
    earth_ratio: float = 0.0   # fraction of brown/earth-tone pixels


@dataclass
class ImageAnalysis:
    """Combined analysis for one image – aggregated by the selector."""

    image_path: str
    quality: QualityResult | None = None
    person_match: PersonMatchResult | None = None
    nature_scene: NatureSceneResult | None = None
    selected: bool = False
    rank: int = 0
    composite_score: float = 0.0  # weighted combination of all criteria
    reason: str = ""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

def _merge_quality(
    existing: dict[str, QualityResult],
    new: dict[str, QualityResult],
) -> dict[str, QualityResult]:
    """Merge quality results – new values overwrite existing per image."""
    return {**existing, **new}


def _merge_person(
    existing: dict[str, PersonMatchResult],
    new: dict[str, PersonMatchResult],
) -> dict[str, PersonMatchResult]:
    return {**existing, **new}


def _merge_nature(
    existing: dict[str, NatureSceneResult],
    new: dict[str, NatureSceneResult],
) -> dict[str, NatureSceneResult]:
    return {**existing, **new}


def _merge_analyses(
    existing: dict[str, ImageAnalysis],
    new: dict[str, ImageAnalysis],
) -> dict[str, ImageAnalysis]:
    return {**existing, **new}


class PhotoSelectorState(TypedDict, total=False):
    """The single state object shared across all graph nodes.

    Fields annotated with a reducer (``Annotated[..., fn]``) are *merged*
    when a node returns a partial update.  Plain fields are overwritten.
    """

    # ---- Inputs (set once before the graph runs) -------------------------
    image_paths: list[str]          # absolute paths to images to analyse
    reference_image_path: str | None  # sample image of the person to find
    select_count: int               # how many images to select (default: 100)

    # ---- Per-agent results (accumulated) ---------------------------------
    quality_results: Annotated[dict[str, QualityResult], _merge_quality]
    person_match_results: Annotated[dict[str, PersonMatchResult], _merge_person]
    nature_scene_results: Annotated[dict[str, NatureSceneResult], _merge_nature]

    # ---- Selector config ---------------------------------------------------
    top_n: int  # select top N images (0 = all above threshold)

    # ---- Selector output --------------------------------------------------
    analyses: Annotated[dict[str, ImageAnalysis], _merge_analyses]
    selected_images: list[str]  # final ranked list of image paths

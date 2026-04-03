"""graph – Sequential pipeline that wires the agents together.

Pipeline
--------
                   START
                    │
                    ▼
             ┌──────────────┐
             │ quality_agent│   ← analyse resolution, blur, noise, artifacts
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │person_match  │   ← DeepFace ArcFace identity matching (if reference provided)
             │    agent     │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │   selector   │   ← deterministic scorer ranks & selects
             └──────┬───────┘
                    │
                    ▼
                   END

All agents run **sequentially** to avoid memory / CPU overload.
Each agent processes one image at a time and frees memory before
moving to the next image.

Note: CLIP model has been removed to save memory. Person matching
is handled by DeepFace ArcFace, and nature scene detection has been removed.

Usage
-----
    from photo_selector.agents.graph import build_graph

    pipeline = build_graph()
    result = pipeline({
        "image_paths": ["/path/to/img1.jpg", "/path/to/img2.png"],
        "reference_image_path": "/path/to/me.jpg",
        "top_n": 100,
    })
    print(result["selected_images"])
"""

from __future__ import annotations

from typing import Any, Callable

from photo_selector.agents.quality_agent import quality_node
from photo_selector.agents.person_match_agent import person_match_node
from photo_selector.agents.selector_agent import selector_node


def build_graph() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Construct a simple sequential pipeline.

    Returns a callable that accepts a state dict and returns the
    updated state after running all agents in order:
      quality_agent → person_match_agent → selector
    """

    def pipeline(state: dict[str, Any]) -> dict[str, Any]:
        # Ensure default values for accumulated result dicts
        state.setdefault("quality_results", {})
        state.setdefault("person_match_results", {})
        state.setdefault("nature_scene_results", {})  # Empty - no longer used
        state.setdefault("analyses", {})
        state.setdefault("selected_images", [])

        # Step 1: Quality analysis (OpenCV / Pillow)
        quality_update = quality_node(state)
        state["quality_results"] = {
            **state["quality_results"],
            **quality_update.get("quality_results", {}),
        }

        # Step 2: Person matching with DeepFace ArcFace (if reference provided)
        person_update = person_match_node(state)
        state["person_match_results"] = {
            **state["person_match_results"],
            **person_update.get("person_match_results", {}),
        }

        # Step 3: Deterministic scoring & selection
        selector_update = selector_node(state)
        state["analyses"] = {
            **state["analyses"],
            **selector_update.get("analyses", {}),
        }
        state["selected_images"] = selector_update.get("selected_images", [])

        return state

    return pipeline

"""cli – Click-based command-line interface for photo-selector."""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from photo_selector import __version__
from photo_selector.scanner import SUPPORTED_EXTENSIONS, scan_directories

console = Console()

# Configure logging to be visible
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="photo-selector")
@click.pass_context
def main(ctx: click.Context) -> None:
    """photo-selector – Scan, list, and process your photos."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument(
    "directories",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recurse into sub-directories (default: yes).",
)
@click.option(
    "--sort-by",
    type=click.Choice(["name", "size", "ext"], case_sensitive=False),
    default="name",
    help="Sort images by name, size, or extension.",
)
def scan(
    directories: tuple[str, ...],
    recursive: bool,
    sort_by: str,
) -> None:
    """Scan one or more DIRECTORIES for images and list them."""
    dir_paths = [Path(d) for d in directories]

    with console.status("[bold green]Scanning directories…[/]"):
        results = scan_directories(dir_paths, recursive=recursive)

    total_images = 0

    for directory, images in results.items():
        if sort_by == "size":
            images.sort(key=lambda img: img.size_bytes, reverse=True)
        elif sort_by == "ext":
            images.sort(key=lambda img: (img.extension, img.name.lower()))
        # default "name" sorting is already done by scanner

        total_images += len(images)

        table = Table(
            title=f"[bold]{directory}[/]",
            title_style="cyan",
            show_lines=False,
        )
        table.add_column("#", justify="right", style="dim", width=5)
        table.add_column("File Name", style="green")
        table.add_column("Extension", justify="center", style="magenta")
        table.add_column("Size", justify="right", style="yellow")
        table.add_column("Path", style="dim")

        for idx, img in enumerate(images, start=1):
            table.add_row(
                str(idx),
                img.name,
                img.extension,
                img.size_human,
                str(img.path.parent),
            )

        console.print()
        if images:
            console.print(table)
        else:
            console.print(
                f"  [dim]No supported images found in {directory}[/]"
            )

    # Summary
    ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    console.print()
    console.print(
        f"[bold]Total:[/] {total_images} image(s) found across "
        f"{len(results)} director{'y' if len(results) == 1 else 'ies'}."
    )
    console.print(f"[dim]Supported formats: {ext_list}[/]")


# -----------------------------------------------------------------------
# process – Run the agent pipeline
# -----------------------------------------------------------------------


@main.command()
@click.argument(
    "directories",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a reference image of YOU. Images containing the same person will score higher.",
)
@click.option(
    "--count",
    "-n",
    type=int,
    default=100,
    show_default=True,
    help="Select the top N images (0 = all above quality threshold).",
)
def process(
    directories: tuple[str, ...],
    reference: str | None,
    count: int,
) -> None:
    """Run the AI agent pipeline on images in DIRECTORIES.

    Scans for images, analyses quality (blur, noise, artifacts), optionally
    matches your face via a reference image (DeepFace ArcFace), then selects
    the top N photos.
    
    Note: CLIP model has been removed to save memory. Person matching is
    handled by DeepFace ArcFace model.
    """
    from photo_selector.agents.graph import build_graph

    # Step 1: Discover images using the existing scanner
    dir_paths = [Path(d) for d in directories]

    with console.status("[bold green]Scanning for images…[/]"):
        scan_results = scan_directories(dir_paths, recursive=True)

    all_paths: list[str] = []
    for images in scan_results.values():
        all_paths.extend(str(img.path) for img in images)

    if not all_paths:
        console.print("[red]No images found in the specified directories.[/]")
        raise SystemExit(1)

    ref_info = f" Reference: [bold]{Path(reference).name}[/]." if reference else ""
    console.print(
        f"[green]Found {len(all_paths)} image(s). "
        f"Selecting top {count if count > 0 else 'all qualifying'}."
        f"{ref_info} "
        f"Running agent pipeline…[/]\n"
    )

    # Step 2: Build and invoke the pipeline in batches
    pipeline = build_graph()
    
    # Process in smaller batches to reduce memory usage
    # Smaller batches = less memory per batch
    # Start with 10, can be adjusted based on available memory
    BATCH_SIZE = 10  # Very small batches to avoid OOM
    total_batches = (len(all_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    console.print(f"[bold cyan]Running agent pipeline in batches of {BATCH_SIZE}…[/]")
    console.print(f"[dim]Processing {len(all_paths)} images in {total_batches} batch(es). This may take a few minutes.[/]\n")
    
    # Accumulate results from all batches
    accumulated_quality_results = {}
    accumulated_person_results = {}
    accumulated_nature_results = {}
    
    # Process each batch
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_paths))
        batch_paths = all_paths[start_idx:end_idx]
        
        console.print(f"[bold]Processing batch {batch_num + 1}/{total_batches} ({len(batch_paths)} images)…[/]")
        
        # Process this batch through quality and person matching only (skip selector)
        from photo_selector.agents.quality_agent import quality_node
        from photo_selector.agents.person_match_agent import person_match_node
        
        batch_state = {
            "image_paths": batch_paths,
            "reference_image_path": reference,
            "quality_results": {},
            "person_match_results": {},
            "nature_scene_results": {},  # Empty - no longer used
        }
        
        # Run quality analysis for this batch
        quality_update = quality_node(batch_state)
        batch_state["quality_results"].update(quality_update.get("quality_results", {}))
        
        # Force GC after quality
        import gc
        gc.collect()
        
        # Run person matching for this batch (DeepFace ArcFace)
        person_update = person_match_node(batch_state)
        batch_state["person_match_results"].update(person_update.get("person_match_results", {}))
        
        batch_result = batch_state
        
        # Accumulate results
        accumulated_quality_results.update(batch_result.get("quality_results", {}))
        accumulated_person_results.update(batch_result.get("person_match_results", {}))
        accumulated_nature_results.update(batch_result.get("nature_scene_results", {}))
        
        # Aggressive memory cleanup after each batch
        import gc
        gc.collect()
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    # Step 3: Run final selection on all accumulated results
    console.print(f"\n[bold]Running final selection on {len(all_paths)} images…[/]")
    
    from photo_selector.agents.selector_agent import selector_node
    
    final_state = {
        "image_paths": all_paths,
        "quality_results": accumulated_quality_results,
        "person_match_results": accumulated_person_results,
        "nature_scene_results": accumulated_nature_results,
        "top_n": count,
    }
    
    selector_result = selector_node(final_state)
    
    # Build final result structure
    result = {
        "selected_images": selector_result.get("selected_images", []),
        "analyses": selector_result.get("analyses", {}),
        "quality_results": accumulated_quality_results,
        "person_match_results": accumulated_person_results,
        "nature_scene_results": accumulated_nature_results,
    }
    
    console.print()  # Add spacing after pipeline completes

    # Step 3: Display results
    selected = result.get("selected_images", [])
    analyses = result.get("analyses", {})

    # Show selected images table (sorted by rank)
    selected_analyses = sorted(
        [a for a in analyses.values() if a.selected],
        key=lambda a: a.rank,
    )

    table = Table(title="[bold]Selected Photos[/]", title_style="cyan", expand=True)
    table.add_column("Rank", justify="right", style="dim", width=5)
    table.add_column("File", style="green", no_wrap=True, min_width=30)
    table.add_column("Quality", justify="center", style="yellow")
    table.add_column("Face", justify="center", style="magenta")
    table.add_column("Nature", justify="center", style="blue")
    table.add_column("Score", justify="center", style="bold")

    for analysis in selected_analyses:
        quality_str = "—"
        person_str = "—"
        nature_str = "—"

        if analysis.quality:
            q = analysis.quality
            sharp = "✓" if q.is_sharp else "✗"
            quality_str = f"{q.overall_quality:.0%} {sharp} ({q.resolution_label})"

        if analysis.person_match:
            p = analysis.person_match
            # Show match status: ✓ if match_found, ✗ if no match, but always show confidence
            match_status = "✓" if p.match_found else "✗"
            person_str = (
                f"{match_status} "
                f"({p.match_confidence:.0%})"
            )
            # Add note if confidence is high but not matching (threshold issue)
            if p.match_confidence > 0.5 and not p.match_found:
                person_str += " [no match]"

        if analysis.nature_scene:
            n = analysis.nature_scene
            nature_str = (
                f"{'✓' if n.is_nature else '✗'} "
                f"({n.nature_confidence:.0%})"
            )
            if n.scene_tags:
                nature_str += f" {', '.join(n.scene_tags[:3])}"

        table.add_row(
            str(analysis.rank),
            Path(analysis.image_path).name,
            quality_str,
            person_str,
            nature_str,
            f"{analysis.composite_score:.0%}",
        )

    console.print(table)

    # Summary
    total_with_faces = sum(
        1 for a in analyses.values()
        if a.person_match and a.person_match.match_found
    )
    console.print()
    console.print(f"[bold]Selected:[/] {len(selected)} of {len(all_paths)} image(s).")
    face_label = "Identity matches" if reference else "Faces detected in"
    console.print(f"[dim]  {face_label}: {total_with_faces} images[/]")
    console.print(f"[dim]  Model:             DeepFace ArcFace[/]")
    if reference:
        console.print(f"[dim]  Reference image:   {Path(reference).name}[/]")

    # Step 4: Copy selected photos to timestamped folder
    if selected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = Path(f"selected_{timestamp}")
        output_folder.mkdir(exist_ok=True)

        with console.status(f"[bold green]Copying {len(selected)} photo(s) to {output_folder}…[/]"):
            for image_path in selected:
                src_path = Path(image_path)
                dst_path = output_folder / src_path.name
                # Handle duplicate filenames by adding a counter
                counter = 1
                while dst_path.exists():
                    stem = src_path.stem
                    suffix = src_path.suffix
                    dst_path = output_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.copy2(src_path, dst_path)

        console.print()
        console.print(f"[bold green]✓[/] Copied {len(selected)} photo(s) to [bold]{output_folder}[/]")


if __name__ == "__main__":
    main()

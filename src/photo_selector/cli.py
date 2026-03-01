"""cli – Click-based command-line interface for photo-selector."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from photo_selector import __version__
from photo_selector.scanner import SUPPORTED_EXTENSIONS, scan_directories

console = Console()


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


if __name__ == "__main__":
    main()

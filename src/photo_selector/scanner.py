"""scanner – Discover and catalogue image files inside one or more directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Supported image extensions (lower-cased).
# Easy to extend when new formats are needed.
SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
}


@dataclass
class ImageInfo:
    """Lightweight metadata container for a discovered image."""

    path: Path
    name: str = field(init=False)
    extension: str = field(init=False)
    size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.path.name
        self.extension = self.path.suffix.lower()
        self.size_bytes = self.path.stat().st_size

    @property
    def size_human(self) -> str:
        """Return a human-readable file size (e.g. '2.4 MB')."""
        size = self.size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


def scan_directory(
    directory: Path,
    *,
    recursive: bool = True,
) -> list[ImageInfo]:
    """Scan *directory* and return an :class:`ImageInfo` for every supported image.

    Parameters
    ----------
    directory:
        Root directory to scan.
    recursive:
        If ``True`` (default), walk into sub-directories.

    Returns
    -------
    list[ImageInfo]
        Sorted alphabetically by file name.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    NotADirectoryError
        If *directory* is not a directory.
    """
    directory = Path(directory).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    images: list[ImageInfo] = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    for entry in iterator:
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(ImageInfo(path=entry))

    images.sort(key=lambda img: img.name.lower())
    return images


def scan_directories(
    directories: list[Path],
    *,
    recursive: bool = True,
) -> dict[Path, list[ImageInfo]]:
    """Scan multiple directories and return results keyed by directory.

    This is the main entry point used by the CLI when the user supplies
    one or more paths.
    """
    results: dict[Path, list[ImageInfo]] = {}
    for d in directories:
        resolved = Path(d).expanduser().resolve()
        results[resolved] = scan_directory(resolved, recursive=recursive)
    return results

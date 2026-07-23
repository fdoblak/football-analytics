"""Deterministic source-frame tiling for small-object (ball) detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class TilingError(ValueError):
    """Tiling / coordinate mapping failure."""


@dataclass(frozen=True)
class TileSpec:
    """Axis-aligned tile in source-frame pixel coordinates [x0,y0,x1,y1)."""

    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    edge_flags: tuple[str, ...]

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)


def _edge_flags(
    *, x0: int, y0: int, x1: int, y1: int, frame_w: int, frame_h: int
) -> tuple[str, ...]:
    flags: list[str] = []
    if x0 <= 0:
        flags.append("left")
    if y0 <= 0:
        flags.append("top")
    if x1 >= frame_w:
        flags.append("right")
    if y1 >= frame_h:
        flags.append("bottom")
    return tuple(flags)


def generate_tiles(
    frame_width: int,
    frame_height: int,
    *,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    max_tiles: int,
) -> list[TileSpec]:
    """Generate a deterministic left-to-right, top-to-bottom tile grid.

    Tiles are clipped to source bounds. ``max_tiles`` is a hard cap; excess tiles
    are dropped (deterministic order preserved).
    """
    if frame_width <= 0 or frame_height <= 0:
        raise TilingError("frame dimensions must be positive")
    if tile_width <= 0 or tile_height <= 0:
        raise TilingError("tile dimensions must be positive")
    if overlap_x < 0 or overlap_y < 0:
        raise TilingError("overlap must be >= 0")
    if overlap_x >= tile_width or overlap_y >= tile_height:
        raise TilingError("overlap must be < tile size")
    if max_tiles < 1:
        raise TilingError("max_tiles must be >= 1")

    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y
    tiles: list[TileSpec] = []
    row = 0
    y = 0
    while y < frame_height:
        col = 0
        x = 0
        y1 = min(y + tile_height, frame_height)
        y0 = max(0, y1 - tile_height) if y1 - y < tile_height and y > 0 else y
        y0 = max(0, min(y0, frame_height - 1))
        y1 = min(frame_height, max(y0 + 1, y1))
        while x < frame_width:
            x1 = min(x + tile_width, frame_width)
            x0 = max(0, x1 - tile_width) if x1 - x < tile_width and x > 0 else x
            x0 = max(0, min(x0, frame_width - 1))
            x1 = min(frame_width, max(x0 + 1, x1))
            tile_id = f"r{row}c{col}"
            tiles.append(
                TileSpec(
                    tile_id=tile_id,
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    edge_flags=_edge_flags(
                        x0=int(x0),
                        y0=int(y0),
                        x1=int(x1),
                        y1=int(y1),
                        frame_w=frame_width,
                        frame_h=frame_height,
                    ),
                )
            )
            if len(tiles) >= max_tiles:
                return tiles
            if x1 >= frame_width:
                break
            x += step_x
            col += 1
        if y1 >= frame_height:
            break
        y += step_y
        row += 1
    return tiles


def map_tile_bbox_to_source(
    bbox: Sequence[float],
    tile: TileSpec,
    *,
    coordinate_space: str = "tile_local",
) -> tuple[float, float, float, float]:
    """Map a bbox into source-frame coordinates.

    Coordinate spaces:
    - ``tile_local``: Ultralytics ran on a cropped tile image; xyxy is relative to
      the tile origin → add ``(tile.x0, tile.y0)``.
    - ``source``: boxes already in full-frame / source space (no offset).
    """
    if coordinate_space not in {"tile_local", "source"}:
        raise TilingError("coordinate_space must be tile_local|source")
    if len(bbox) != 4:
        raise TilingError("bbox must be length-4 xyxy")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if coordinate_space == "source":
        return (x1, y1, x2, y2)
    return (x1 + float(tile.x0), y1 + float(tile.y0), x2 + float(tile.x0), y2 + float(tile.y0))


def crop_tile(frame_bgr: Any, tile: TileSpec) -> Any:
    """Return a view/copy of ``frame_bgr[y0:y1, x0:x1]``."""
    return frame_bgr[tile.y0 : tile.y1, tile.x0 : tile.x1]


__all__ = [
    "TilingError",
    "TileSpec",
    "generate_tiles",
    "map_tile_bbox_to_source",
    "crop_tile",
]

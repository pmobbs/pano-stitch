#!/usr/bin/env python3
"""CLI entrypoint for notebook-style panorama stitching."""

import argparse
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage import io

from p2Helpers import findTemplateLocation


Location = Tuple[int, int]
Edge = Dict[str, object]


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize image shape to HxWx3."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def load_images(image_paths: Sequence[str]) -> Tuple[List[np.ndarray], List[str]]:
    images: List[np.ndarray] = []
    names: List[str] = []

    for path in image_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(_to_rgb(io.imread(str(p))))
        names.append(p.stem)

    return images, names


def choose_template(img_a: np.ndarray, img_b: np.ndarray, tile_factor: int) -> Optional[np.ndarray]:
    """Return best tile from img_a found in img_b, mirroring notebook logic."""
    best_scores: List[Tuple[Location, float, int, int]] = []

    tile_w = img_a.shape[0] // tile_factor
    tile_h = img_a.shape[1] // tile_factor
    if tile_w <= 1 or tile_h <= 1:
        return None

    for i in range(tile_factor):
        for j in range(tile_factor):
            xb = i * tile_w
            yb = j * tile_h
            xf = (i + 1) * tile_w - 1
            yf = (j + 1) * tile_h - 1

            tile = img_a[xb:xf, yb:yf]
            if tile.size == 0:
                continue

            _, x1, x2, y1, y2, score, corr, _ = findTemplateLocation(img_b, tile)
            if score != -1:
                best_scores.append(((x1, y1), float(corr), x2 - x1, y2 - y1))

    if not best_scores:
        return None

    # Same selection strategy as notebook: highest correlation among candidates.
    location, _, h, w = max(best_scores, key=lambda entry: entry[1])
    x, y = location
    return img_b[x : x + h, y : y + w]


def generate_pair_templates(images: Sequence[np.ndarray], names: Sequence[str], tile_factor: int) -> List[Edge]:
    edges: List[Edge] = []
    for i, j in combinations(range(len(images)), 2):
        template = choose_template(images[i], images[j], tile_factor=tile_factor)
        if template is None or template.size == 0:
            continue
        edges.append(
            {
                "pair": (i, j),
                "pair_name": f"{names[i]}-{names[j]}",
                "template": template,
            }
        )
    return edges


def locate_templates(images: Sequence[np.ndarray], edges: List[Edge]) -> List[Edge]:
    valid_edges: List[Edge] = []

    for edge in edges:
        i, j = edge["pair"]  # type: ignore[misc]
        template = edge["template"]  # type: ignore[misc]

        cp_i, xb_i, _, yb_i, _, score_i, _, _ = findTemplateLocation(images[i], template)
        cp_j, xb_j, _, yb_j, _, score_j, _, _ = findTemplateLocation(images[j], template)

        if cp_i == -1 or cp_j == -1:
            continue

        edge["loc_i"] = (int(xb_i), int(yb_i))
        edge["loc_j"] = (int(xb_j), int(yb_j))
        edge["score_i"] = float(score_i)
        edge["score_j"] = float(score_j)
        valid_edges.append(edge)

    return valid_edges


def solve_positions(images: Sequence[np.ndarray], edges: Sequence[Edge], margin: int = 40) -> Dict[int, Location]:
    """
    Solve image top-left coordinates using pairwise template offsets.
    For an edge (i,j): pos[j] = pos[i] + (loc_i - loc_j).
    """
    adjacency: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    for edge in edges:
        i, j = edge["pair"]  # type: ignore[misc]
        xi, yi = edge["loc_i"]  # type: ignore[misc]
        xj, yj = edge["loc_j"]  # type: ignore[misc]

        dx = xi - xj
        dy = yi - yj
        adjacency[i].append((j, dx, dy))
        adjacency[j].append((i, -dx, -dy))

    positions: Dict[int, Location] = {}
    visited = set()

    component_anchor_x = 0
    for start in range(len(images)):
        if start in visited:
            continue

        if start not in adjacency:
            positions[start] = (component_anchor_x, 0)
            component_anchor_x += images[start].shape[1] + margin
            visited.add(start)
            continue

        positions[start] = (component_anchor_x, 0)
        queue: deque[int] = deque([start])
        visited.add(start)

        component_nodes = {start}
        while queue:
            node = queue.popleft()
            x0, y0 = positions[node]
            for nbr, dx, dy in adjacency[node]:
                candidate = (x0 + dx, y0 + dy)
                if nbr not in positions:
                    positions[nbr] = candidate
                if nbr not in visited:
                    visited.add(nbr)
                    component_nodes.add(nbr)
                    queue.append(nbr)

        max_y = max(positions[idx][1] + images[idx].shape[1] for idx in component_nodes)
        component_anchor_x = max_y + margin

    return positions


def compose_panorama(images: Sequence[np.ndarray], positions: Dict[int, Location], background: int = 255) -> np.ndarray:
    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())

    max_x = max(x + images[idx].shape[0] for idx, (x, _) in positions.items())
    max_y = max(y + images[idx].shape[1] for idx, (_, y) in positions.items())

    canvas_h = max_x - min_x
    canvas_w = max_y - min_y

    dtype = images[0].dtype
    panorama = np.full((canvas_h, canvas_w, 3), background, dtype=dtype)

    for idx, image in enumerate(images):
        x, y = positions[idx]
        xs = x - min_x
        ys = y - min_y
        xe = xs + image.shape[0]
        ye = ys + image.shape[1]
        panorama[xs:xe, ys:ye, :] = image[:, :, :3]

    return panorama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch images into a panorama using notebook-derived template matching."
    )
    parser.add_argument("images", nargs="+", help="Input image paths in preferred stitching order.")
    parser.add_argument("-o", "--output", required=True, help="Output panorama file path.")
    parser.add_argument(
        "--tile-factor",
        type=int,
        default=4,
        help="Tile split factor for candidate template extraction (default: 4).",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=255,
        help="Background fill value for empty canvas regions (default: 255).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.images) < 2:
        raise ValueError("Provide at least two images to stitch.")
    if args.tile_factor < 2:
        raise ValueError("--tile-factor must be >= 2.")

    images, names = load_images(args.images)
    pair_templates = generate_pair_templates(images, names, tile_factor=args.tile_factor)
    if not pair_templates:
        raise RuntimeError("No candidate templates found between image pairs.")

    valid_edges = locate_templates(images, pair_templates)
    if not valid_edges:
        raise RuntimeError("No valid pairwise alignments found. Try a different image set/order.")

    positions = solve_positions(images, valid_edges)
    panorama = compose_panorama(images, positions, background=args.background)
    io.imsave(args.output, panorama)

    print(f"Saved panorama to: {args.output}")
    print(f"Input images: {len(images)}")
    print(f"Successful pairwise alignments: {len(valid_edges)}")


if __name__ == "__main__":
    main()

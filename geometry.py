"""Target geometry in map coordinates, independent of Qt and PPO.

All shapes must fit inside [0, width] x [0, height]. Ellipses and
rectangles are axis-aligned. The two rectangle/square points are opposite
corners; either order is accepted. Coverage uses the centres of 1 x 1 cells.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np

SHAPES = ("circle", "ellipse", "triangle", "rectangle", "square")


def default_shape_parameters(shape, width, height, scale=1.0):
    """Legacy-compatible defaults; the GUI supplies explicit parameters instead."""
    cx, cy = width / 2, height / 2
    if shape == "circle":
        return {"center": [cx, cy], "radius": 0.25 * min(width, height) * scale}
    if shape == "ellipse":
        return {"center": [cx, cy], "radii": [0.30 * width * scale, 0.20 * height * scale]}
    if shape == "triangle":
        return {"vertices": [[cx + dx * width * scale, cy + dy * height * scale]
                             for dx, dy in ((-0.3, -0.3), (0.3, -0.3), (0.0, 0.3))]}
    if shape in ("rectangle", "square"):
        hx, hy = 0.3 * width * scale, 0.2 * height * scale
        if shape == "square":
            hx = hy = 0.25 * min(width, height) * scale
        return {"point1": [cx - hx, cy - hy], "point2": [cx + hx, cy + hy]}
    raise ValueError(f"Unknown target shape: {shape!r}")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _point(value, name):
    try:
        if len(value) != 2:
            raise ValueError
        return [_number(value[0], f"{name} x"), _number(value[1], f"{name} y")]
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"{name} must contain two finite coordinates (x, y).") from None


def validate_shape_parameters(shape, parameters, width, height):
    """Return normalized, independent, JSON-safe parameters or raise ValueError."""
    if shape not in SHAPES:
        raise ValueError("Target shape must be circle, ellipse, triangle, rectangle or square.")
    if not isinstance(parameters, Mapping):
        raise ValueError("Target parameters must be a mapping.")
    try:
        if shape in ("circle", "ellipse"):
            cx, cy = _point(parameters["center"], "Center")
            if shape == "circle":
                rx = ry = _number(parameters["radius"], "Radius")
                result = {"center": [cx, cy], "radius": rx}
            else:
                rx, ry = _point(parameters["radii"], "Radii (X, Y)")
                result = {"center": [cx, cy], "radii": [rx, ry]}
            if rx <= 0 or ry <= 0:
                raise ValueError("Radius / both ellipse radii must be greater than zero.")
            xmin, xmax, ymin, ymax = cx - rx, cx + rx, cy - ry, cy + ry
        elif shape == "triangle":
            values = parameters["vertices"]
            if len(values) != 3:
                raise ValueError("A triangle requires exactly three points.")
            vertices = [_point(p, f"Point {i + 1}") for i, p in enumerate(values)]
            a, b, c = np.asarray(vertices, dtype=float)
            u, v = b - a, c - a
            cross = u[0] * v[1] - u[1] * v[0]
            scale2 = max(float(u @ u), float(v @ v), float((c-b) @ (c-b)), 1.0)
            if abs(cross) <= 1e-10 * scale2:
                raise ValueError("Triangle points must be distinct and must not lie on one line.")
            result = {"vertices": vertices}
            xs, ys = np.asarray(vertices, dtype=float).T
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        else:
            p1 = _point(parameters["point1"], "Point 1")
            p2 = _point(parameters["point2"], "Point 2")
            w, h = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            if w <= 0 or h <= 0:
                raise ValueError("Opposite corners must define a positive width and height.")
            if shape == "square" and not math.isclose(w, h, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("For a square, |x2 - x1| must equal |y2 - y1|. "
                                 "Use Rectangle for unequal side lengths.")
            result = {"point1": p1, "point2": p2}
            xmin, xmax = sorted((p1[0], p2[0]))
            ymin, ymax = sorted((p1[1], p2[1]))
    except KeyError as exc:
        raise ValueError(f"Missing {shape} parameter: {exc.args[0]}") from None
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {shape}.") from exc
    tolerance = 1e-9
    if (xmin < -tolerance or ymin < -tolerance or xmax > width + tolerance
            or ymax > height + tolerance):
        raise ValueError(f"The entire {shape} must fit inside the map: "
                         f"0 <= x <= {width}, 0 <= y <= {height}. "
                         "Adjust its coordinates/radii or increase the map size.")
    return result


def target_mask(shape, parameters, X, Y):
    """Test cell centres; points on the boundary are included."""
    if shape == "circle":
        cx, cy = parameters["center"]
        return (X - cx)**2 + (Y - cy)**2 <= parameters["radius"]**2
    if shape == "ellipse":
        cx, cy = parameters["center"]
        rx, ry = parameters["radii"]
        return ((X - cx) / rx)**2 + ((Y - cy) / ry)**2 <= 1.0
    if shape == "triangle":
        vertices = np.asarray(parameters["vertices"], dtype=float)
        crosses = []
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            crosses.append((b[0]-a[0]) * (Y-a[1]) - (b[1]-a[1]) * (X-a[0]))
        tolerance = 1e-9
        return (np.logical_and.reduce([c >= -tolerance for c in crosses]) |
                np.logical_and.reduce([c <= tolerance for c in crosses]))
    p1, p2 = parameters["point1"], parameters["point2"]
    xmin, xmax = sorted((p1[0], p2[0]))
    ymin, ymax = sorted((p1[1], p2[1]))
    return (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)


def outline_points(shape, parameters):
    """Closed target outline used only for display (not coverage calculation)."""
    if shape in ("circle", "ellipse"):
        angle = np.linspace(0, 2 * np.pi, 361)
        cx, cy = parameters["center"]
        rx, ry = ((parameters["radius"],) * 2 if shape == "circle"
                  else parameters["radii"])
        return np.column_stack([cx + rx * np.cos(angle), cy + ry * np.sin(angle)])
    if shape == "triangle":
        vertices = np.asarray(parameters["vertices"], dtype=float)
    else:
        p1, p2 = parameters["point1"], parameters["point2"]
        xmin, xmax = sorted((p1[0], p2[0]))
        ymin, ymax = sorted((p1[1], p2[1]))
        vertices = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return np.vstack([vertices, vertices[0]])

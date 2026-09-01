"""Grid geometry: mask -> quadrilateral -> rectified crop.

Pure OpenCV, and deliberately independent of whatever produced the mask.  The
grid detector is a YOLOv8n segmentation model, but nothing here knows that:
these helpers take a binary mask (or a quad) and turn it into corners and a
perspective warp, so a different detector can be swapped in without touching
any of it.
"""

from __future__ import annotations

import cv2
import numpy as np

# How far from each mask edge to search for the real grid border, as a fraction
# of the grid extent, when `refine_on_grid_lines` is used.  Kept wide because it
# was tuned for a detector that under-segmented the bottom edge; the YOLO
# detector does not need it, and enabling it there hurt badly (mean corner error
# 5.7 % -> 40.9 % on 100 held-out images).
BAND_OUT = 0.14
BAND_IN = 0.04


def seg_overlay(
    image: np.ndarray, mask: np.ndarray, corners: np.ndarray
) -> np.ndarray:
    """Draw the segmentation mask and corner quadrilateral on image."""
    vis = image.astype(np.float32)

    # Teal tint over masked region
    teal = np.zeros_like(vis)
    teal[mask > 0] = [30, 215, 160]
    alpha = np.where(mask[:, :, None] > 0, 0.35, 0.0)
    vis = (vis * (1 - alpha) + teal * alpha).clip(0, 255).astype(np.uint8)

    # Quadrilateral border
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (30, 220, 130), 3, cv2.LINE_AA)

    # Corner markers: white ring + green fill
    for pt in corners.astype(np.int32):
        cv2.circle(vis, tuple(pt), 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pt), 6, (20, 160, 80), -1, cv2.LINE_AA)

    return vis

# ------------------------------------------------------------------
# Mask-based corner detection
# ------------------------------------------------------------------

def mask_contour(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour of the segmentation mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def corners_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Fit a quadrilateral to the segmentation mask by fitting a line to
    each of its four sides, then intersecting adjacent lines.

    Fitting lines beats simplifying the outline with approxPolyDP: a ragged
    mask boundary averages out over the hundreds of points along each side,
    and four independently fitted lines can express a perspective-skewed
    grid, which a rotated rectangle cannot.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)

    # A rotated rect gives the rough orientation; each boundary point is
    # then assigned to whichever of its four sides it lies nearest.
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    dist = np.zeros((len(pts), 4), np.float32)
    for i, (a, b) in enumerate(sides):
        ab = b - a
        L = np.hypot(*ab) + 1e-6
        dist[:, i] = np.abs(np.cross(np.broadcast_to(ab, (len(pts), 2)), pts - a)) / L
    owner = dist.argmin(1)

    lines = []
    for i, (a, b) in enumerate(sides):
        sel = pts[owner == i]
        if len(sel) < 10:
            return None
        # Drop points near the two ends: mask corners are rounded, and
        # including them bends the fitted side inwards.
        ab = b - a
        L = np.hypot(*ab) + 1e-6
        u = ab / L
        t = (sel - a) @ u
        keep = (t > 0.12 * L) & (t < 0.88 * L)
        if keep.sum() >= 10:
            sel = sel[keep]
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((float(x0), float(y0), float(vx), float(vy)))
    return quad_from_lines(lines)

def quad_from_lines(lines: list[tuple]) -> np.ndarray | None:
    """Intersect four consecutive (point, direction) lines into a quad."""
    def intersect(l1, l2):
        x1, y1, vx1, vy1 = l1
        x2, y2, vx2, vy2 = l2
        den = vx1 * vy2 - vy1 * vx2
        if abs(den) < 1e-9:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
        return (x1 + t * vx1, y1 + t * vy1)

    quad = []
    for i in range(4):
        q = intersect(lines[i], lines[(i + 1) % 4])
        if q is None:
            return None
        quad.append(q)
    return np.array(quad, dtype=np.float32)

def refine_on_grid_lines(
    image: np.ndarray,
    mask: np.ndarray,
    quad: np.ndarray,
    band_out: float = BAND_OUT,
    band_in: float = BAND_IN,
    ang_tol: float = 18.0,
) -> np.ndarray:
    """Snap each edge of the mask quad onto the real grid border near it.

    The mask locates the grid reliably but its *boundary* is not precise --
    a segmentation model may under-segment the bottom edge, cutting
    off the last row.  So the mask decides which four borders we are looking
    for and roughly where they are, and Hough supplies each exact line,
    searched only within a narrow band around that mask edge and only among
    segments of matching orientation.  Nothing is selected merely for being
    outermost, so a page edge or table rule away from the mask cannot win.
    """
    extent = float(np.sqrt(np.count_nonzero(mask)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100, apertureSize=3)
    min_len = max(8, int(extent * 0.25))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, min_len // 2,
                           minLineLength=min_len, maxLineGap=min_len // 3)
    if segs is None:
        return quad
    segs = segs.reshape(-1, 4).astype(np.float32)
    mid = np.stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2], 1)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])

    centre = quad.mean(0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = np.hypot(*ab) + 1e-6
        normal = np.array([-ab[1], ab[0]]) / L
        if np.dot(normal, centre - (a + b) / 2) > 0:      # point it outward
            normal = -normal
        edge_ang = np.degrees(np.arctan2(ab[1], ab[0])) % 180
        d_ang = np.abs(((ang - edge_ang + 90) % 180) - 90)
        offset = (mid - (a + b) / 2) @ normal              # +ve = outside edge
        ok = ((d_ang < ang_tol)
              & (offset > -band_in * extent)
              & (offset < band_out * extent)
              & (length > 0.30 * L))
        if not ok.any():
            lines.append((float(a[0]), float(a[1]), float(ab[0] / L), float(ab[1] / L)))
            continue
        cand = np.where(ok)[0]
        # Longest segment wins, with a mild preference for the outer one:
        # the true border is the longest line in the band, and ties should
        # resolve outwards rather than onto the first inner grid line.
        best = cand[np.argmax(length[cand] / extent + 0.25 * offset[cand] / extent)]
        pair = np.array([[segs[best, 0], segs[best, 1]],
                         [segs[best, 2], segs[best, 3]]], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pair, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((float(x0), float(y0), float(vx), float(vy)))

    refined = quad_from_lines(lines)
    if refined is None:
        return quad
    area = abs(cv2.contourArea(refined))
    mask_area = float(np.count_nonzero(mask))
    if mask_area <= 0 or not (0.7 <= area / mask_area <= 1.8):
        return quad
    return refined

# ------------------------------------------------------------------
# Quad detection fallback (mask contour)
# ------------------------------------------------------------------

def find_quad(contour: np.ndarray) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32)

    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    s = hull.sum(axis=1)
    d = hull[:, 0] - hull[:, 1]
    return np.array([
        hull[np.argmin(s)],
        hull[np.argmax(d)],
        hull[np.argmax(s)],
        hull[np.argmin(d)],
    ], dtype=np.float32)

# ------------------------------------------------------------------
# Perspective warp
# ------------------------------------------------------------------

def perspective_warp(image: np.ndarray, quad: np.ndarray, size: int = 450) -> np.ndarray:
    pts = quad.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left

    # 2 % outward expansion absorbs the line half-width so that outer
    # border pixels land inside the output rather than exactly on edge.
    center = rect.mean(axis=0)
    rect = center + 1.02 * (rect - center)

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(rect, dst)
    return cv2.warpPerspective(image, H, (size, size))

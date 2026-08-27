import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .config import GridDetectorConfig


class GridDetector:
    """Detect sudoku grids and apply perspective correction via Mask R-CNN + homography.

    Pipeline:
      1. Mask R-CNN locates the approximate grid region.
      2. Probabilistic Hough line transform within the expanded mask bounding box
         finds the actual outer-border line equations (long lines that span ≥ 40 %
         of the region; short digit strokes are discarded).
      3. The four corner points are the intersections of the outermost H×V line
         pairs, giving a true quadrilateral warp even for perspective-distorted grids.
    """

    def __init__(self, config: GridDetectorConfig | None = None, device: str | None = None):
        self.cfg = config or GridDetectorConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"Mask R-CNN weights not found: {self.cfg.model_path}")
        self.model = self._build_and_load_model()
        self.model.eval()

    def _build_and_load_model(self):
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
        model.load_state_dict(
            torch.load(self.cfg.model_path, map_location=self.device, weights_only=False)
        )
        model.to(self.device)
        return model

    def _run(self, image: np.ndarray):
        """Run full detection; return (rectified, resized, mask_np, corners)."""
        resized = cv2.resize(image, self.cfg.resize_to)

        img_tensor = transforms.ToTensor()(resized).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_tensor])

        masks = outputs[0]["masks"][outputs[0]["scores"] > self.cfg.detection_threshold]
        if masks.shape[0] == 0:
            raise RuntimeError("No sudoku grid detected. Try a clearer image or lower threshold.")

        h_img, w_img = resized.shape[:2]
        combined = torch.zeros((h_img, w_img), dtype=torch.uint8, device=self.device)
        for m in masks:
            combined = torch.max(combined, (m[0] > 0.5).byte())
        mask_np = (combined.cpu().numpy() * 255).astype(np.uint8)

        corners = self._corners_from_hough(resized, mask_np)
        if corners is None:
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise RuntimeError("No contour found from detection mask.")
            contour = max(contours, key=cv2.contourArea)
            corners = self._find_quad(contour)

        rectified = self._perspective_warp(resized, corners, size=self.cfg.output_size)
        return rectified, resized, mask_np, corners

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect sudoku grid and return perspective-corrected crop.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Rectified square grid image of size (output_size, output_size, 3).
        """
        rectified, _, _, _ = self._run(image)
        return rectified

    def detect_debug(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Like detect() but also returns a segmentation overlay for visualization.

        Returns:
            (rectified, seg_overlay) — rectified grid and the mask+corner
            overlay drawn on the resized source image.
        """
        rectified, resized, mask_np, corners = self._run(image)
        seg = self._seg_overlay(resized, mask_np, corners)
        return rectified, seg

    @staticmethod
    def _seg_overlay(
        image: np.ndarray, mask: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Draw the MaskRCNN segmentation mask and corner quadrilateral on image."""
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
    # Hough-based corner detection
    # ------------------------------------------------------------------

    @staticmethod
    def _corners_from_hough(
        image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray | None:
        """Find the 4 outer-border corners via Hough line detection.

        Within the mask bounding box (expanded 15 % on each side) we run
        probabilistic Hough on Canny edges.  We keep only lines that span at
        least 40 % of the crop dimension in their dominant direction, so digit
        strokes (~5 % of image) are eliminated while grid lines (≥ 40 %) survive.

        The outermost detected horizontal line pair (top, bottom) and vertical
        line pair (left, right) are intersected to give 4 true corner points.
        """
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, bw, bh = cv2.boundingRect(coords)

        pad = int(max(bw, bh) * 0.15)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + bw + pad)
        y1 = min(image.shape[0], y + bh + pad)

        crop_gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        ch, cw = crop_gray.shape

        blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

        # Minimum line segment to keep: 30 % of the smaller crop dimension.
        # This filters digit strokes (typically ≤ 10 %) while retaining grid
        # lines that span most of the image.
        min_segment = int(min(cw, ch) * 0.30)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,         # 0.5° resolution
            threshold=min_segment // 2,
            minLineLength=min_segment,
            maxLineGap=min_segment // 4,
        )
        if lines is None or len(lines) < 4:
            return None

        # Minimum projected span along dominant axis for a "long" grid line.
        min_h_span = cw * 0.40   # horizontal line must span ≥ 40 % of width
        min_v_span = ch * 0.40   # vertical line must span ≥ 40 % of height

        h_lines: list[tuple] = []   # (x1,y1,x2,y2, y_center)
        v_lines: list[tuple] = []   # (x1,y1,x2,y2, x_center)

        for seg in lines:
            lx1, ly1, lx2, ly2 = seg if seg.ndim == 1 else seg[0]
            dx, dy = lx2 - lx1, ly2 - ly1
            angle = abs(np.degrees(np.arctan2(dy, dx))) % 180

            if angle < 20 or angle > 160:          # near-horizontal
                if abs(dx) >= min_h_span:
                    h_lines.append((lx1, ly1, lx2, ly2, (ly1 + ly2) / 2))
            elif 70 < angle < 110:                  # near-vertical
                if abs(dy) >= min_v_span:
                    v_lines.append((lx1, ly1, lx2, ly2, (lx1 + lx2) / 2))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        top_line = min(h_lines, key=lambda l: l[4])
        bot_line = max(h_lines, key=lambda l: l[4])
        lft_line = min(v_lines, key=lambda l: l[4])
        rgt_line = max(v_lines, key=lambda l: l[4])

        def intersect(
            la: tuple, lb: tuple
        ) -> tuple[float, float] | None:
            x1_, y1_, x2_, y2_ = la[:4]
            x3_, y3_, x4_, y4_ = lb[:4]
            denom = (x1_ - x2_) * (y3_ - y4_) - (y1_ - y2_) * (x3_ - x4_)
            if abs(denom) < 1e-6:
                return None
            t = ((x1_ - x3_) * (y3_ - y4_) - (y1_ - y3_) * (x3_ - x4_)) / denom
            return x1_ + t * (x2_ - x1_), y1_ + t * (y2_ - y1_)

        tl = intersect(top_line, lft_line)
        tr = intersect(top_line, rgt_line)
        br = intersect(bot_line, rgt_line)
        bl = intersect(bot_line, lft_line)

        if any(p is None for p in (tl, tr, br, bl)):
            return None

        return np.array([
            [x0 + tl[0], y0 + tl[1]],
            [x0 + tr[0], y0 + tr[1]],
            [x0 + br[0], y0 + br[1]],
            [x0 + bl[0], y0 + bl[1]],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Quad detection fallback (mask contour)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_quad(contour: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _perspective_warp(image: np.ndarray, quad: np.ndarray, size: int = 450) -> np.ndarray:
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

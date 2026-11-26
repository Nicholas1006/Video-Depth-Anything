"""
Measurements from a parallel stereo pair with a known baseline.

Assumed setup: two identical cameras side by side, pointing the same way and
fixed `baseline` metres apart (a rigid rig, so every frame pair is already
aligned). A scene point then appears on the same row in both images, shifted
horizontally by its disparity d = x_left - x_right (pixels). From d and the
baseline B alone:

    metres per pixel at that point            m = B / d
    size of something N pixels across          N * B / d
    offset from the camera's optical axis      (u - cx) * B / d ,  (v - cy) * B / d

Distance along the viewing direction also needs the focal length f in pixels:

    Z = f * B / d

Why it cannot be avoided: a scene filmed with focal length f, and the same
scene stretched to twice the depth filmed with 2f, give pixel-identical image
pairs (u = fX/Z = 2fX/2Z, d = fB/Z = 2fB/2Z). Sizes facing the camera are the
same in both, so they are measurable; distance is not. For a fixed rig f only
has to be found once, e.g. from one object at a measured distance
(focal_from_known_distance), or from the field of view (focal_from_hfov).
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
NAN = float('nan')


# --------------------------------------------------------------------------- #
# Inputs / focal length helpers
# --------------------------------------------------------------------------- #
def load_frame(path: str, frame_index: int = 0) -> np.ndarray:
    """Read an image, or frame `frame_index` of a video, as a BGR uint8 array."""
    if os.path.splitext(path)[1].lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f'could not read frame {frame_index} of {path}')
        return frame
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # handles non-ASCII paths
    if img is None:
        raise ValueError(f'could not decode image {path}')
    return img


def split_sbs(frame: np.ndarray, half: bool = False):
    """Split a side-by-side 3D frame into (left, right) views.

    half=True for 'half SBS', where each view was squeezed to half width: they are stretched back.
    """
    h, w = frame.shape[:2]
    left, right = frame[:, :w // 2], frame[:, w // 2:2 * (w // 2)]
    if half:
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_CUBIC)
    return left, right


def estimate_disparity_offset(pairs, percentile: float = 1.0, max_row_diff_px: float = 1.0) -> float:
    """Estimate a constant horizontal shift added to one view (common in 3D videos made for viewing).

    Assumes the farthest things in view (skyline, distant buildings or hills) are effectively at
    infinity, i.e. should shift ~0 px. Returns the offset to pass as `disparity_offset_px`.
    `pairs` is a list of (left, right) images; using frames spread over a clip is more reliable.
    """
    dx_all = []
    for left, right in pairs:
        pl, pr = match_features(_to_gray(left), _to_gray(right))
        if len(pl):
            on_row = np.abs(pl[:, 1] - pr[:, 1]) <= max_row_diff_px
            dx_all.append(pl[on_row, 0] - pr[on_row, 0])
    dx = np.concatenate(dx_all) if dx_all else np.empty(0)
    if len(dx) < 50:
        raise RuntimeError('too few matches to estimate the disparity offset')
    return float(-np.percentile(dx, percentile))


def focal_from_hfov(image_width: int, hfov_deg: float) -> float:
    """Focal length in pixels from the horizontal field of view."""
    return 0.5 * image_width / np.tan(np.radians(hfov_deg) / 2)


def focal_from_known_distance(disparity_px: float, depth_m: float, baseline_m: float) -> float:
    """Focal length in pixels from one point whose distance along the viewing axis is known.

    Valid for every later shot from the same rig at the same resolution.
    """
    return disparity_px * depth_m / baseline_m


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Measurement:
    box: Tuple[float, float, float, float]
    disparity_px: float                    # median horizontal shift between the two shots
    disparity_iqr_px: Tuple[float, float]  # 25th-75th percentile in the region
    n_samples: int
    source: str                            # 'sparse' (matched features) or 'dense' (SGBM)
    metres_per_px: float                   # B / d: real size of one pixel at that distance
    rel_uncertainty: float                 # +- fraction on sizes and distance from a 0.5 px disparity error
    offset_m: Tuple[float, float]          # region centre: metres right of / below the left camera's axis
    size_m: Optional[Tuple[float, float]] = None   # box width, height (box queries)
    depth_m: Optional[float] = None        # distance along the viewing axis (needs focal length)
    range_m: Optional[float] = None        # straight-line distance from the left camera (needs focal length)
    sparse_disparity_px: float = NAN
    n_sparse: int = 0
    dense_disparity_px: float = NAN
    n_dense: int = 0

    @property
    def mixed_depths(self) -> bool:
        """True when the disparities in the region spread by more than 10%."""
        return (self.disparity_iqr_px[1] - self.disparity_iqr_px[0]) > 0.1 * self.disparity_px

    def __str__(self):
        x0, y0, x1, y1 = self.box
        s = (f'region ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}): shift {self.disparity_px:.2f} px '
             f'from {self.n_samples} {self.source} samples -> {1000 * self.metres_per_px:.3f} mm/px '
             f'(+-{100 * self.rel_uncertainty:.1f}%)')
        if self.size_m is not None:
            s += f'\n    size {self.size_m[0]:.3f} x {self.size_m[1]:.3f} m (w x h)'
        s += f'\n    centre is {self.offset_m[0]:+.3f} m right, {self.offset_m[1]:+.3f} m down from the camera axis'
        if self.depth_m is not None:
            s += f'\n    distance {self.depth_m:.3f} m ahead, {self.range_m:.3f} m in a straight line'
        else:
            s += '\n    distance: needs the focal length (--focal / --hfov, or find it once with --calibrate)'
        if self.source == 'sparse' and self.n_dense:
            s += f'\n    [dense estimate: {self.dense_disparity_px:.2f} px]'
        elif self.source == 'dense' and self.n_sparse:
            s += f'\n    [sparse estimate: {self.sparse_disparity_px:.2f} px from {self.n_sparse} pts]'
        if self.mixed_depths:
            s += ('\n    WARNING: shifts in this region disagree a lot - it probably mixes the object with '
                  'background (or SGBM bled across an edge); tighten the box')
        return s


@dataclass
class StereoResult:
    baseline_m: float
    image_size: Tuple[int, int]            # (w, h)
    principal_point: Tuple[float, float]   # (cx, cy), default image centre
    pts_left: np.ndarray                   # (N, 2) matched points, pixels in the left image
    pts_right: np.ndarray                  # (N, 2) the same points in the right image
    disparity: np.ndarray                  # (N,) horizontal shift, pixels
    n_matches: int                         # raw matches before the same-row check
    row_offset_px: float                   # median vertical offset of kept matches (~0 if aligned)
    focal_px: Optional[float] = None
    disparity_map: Optional[np.ndarray] = None   # (h, w) pixels, NaN where unknown; aligned with left
    warnings: List[str] = field(default_factory=list)

    @property
    def metres_per_px(self) -> np.ndarray:
        return self.baseline_m / self.disparity

    def points_3d(self) -> np.ndarray:
        """(N, 3) metres in the left camera frame (x right, y down, z ahead). Z is NaN without a focal length."""
        m = self.metres_per_px
        cx, cy = self.principal_point
        z = self.focal_px * m if self.focal_px else np.full_like(m, np.nan)
        return np.stack([(self.pts_left[:, 0] - cx) * m, (self.pts_left[:, 1] - cy) * m, z], axis=1)

    @property
    def depth_map(self) -> Optional[np.ndarray]:
        if self.disparity_map is None or not self.focal_px:
            return None
        return self.focal_px * self.baseline_m / self.disparity_map

    def measure(self, point: Optional[Sequence[float]] = None,
                box: Optional[Sequence[float]] = None,
                radius: Optional[float] = None, min_sparse: int = 5,
                disparity_sigma_px: float = 0.5) -> Measurement:
        """Measure whatever is at `point` (x, y) or inside `box` (x0, y0, x1, y1), left-image pixels.

        Uses matched features in the region if there are at least `min_sparse`,
        otherwise the dense disparity map. Medians are used, so keep boxes tight
        around the object: background inside the box drags the estimate.
        """
        w, h = self.image_size
        if box is not None:
            x0, y0, x1, y1 = box
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
        elif point is not None:
            r = radius if radius is not None else max(4, round(0.01 * max(w, h)))
            x0, y0, x1, y1 = point[0] - r, point[1] - r, point[0] + r, point[1] + r
        else:
            raise ValueError('give a point or a box')
        x0, x1 = (float(v) for v in np.clip([x0, x1], 0, w - 1))
        y0, y1 = (float(v) for v in np.clip([y0, y1], 0, h - 1))

        p = self.pts_left
        inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        d_sparse = self.disparity[inside]
        d_dense = np.empty(0)
        if self.disparity_map is not None:
            region = self.disparity_map[int(round(y0)):int(round(y1)) + 1, int(round(x0)):int(round(x1)) + 1]
            d_dense = region[np.isfinite(region)]

        if len(d_sparse) >= min_sparse or (len(d_sparse) and not d_dense.size):
            source, d = 'sparse', d_sparse
        elif d_dense.size:
            source, d = 'dense', d_dense
        else:
            raise RuntimeError('no stereo match in this region (no matched features and no dense '
                               'disparity) - try a larger box or a more textured part of the object')

        disp = float(np.median(d))
        q25, q75 = np.percentile(d, [25, 75])
        m = self.baseline_m / disp
        cx, cy = self.principal_point
        off = (((x0 + x1) / 2 - cx) * m, ((y0 + y1) / 2 - cy) * m)
        depth = rng = None
        if self.focal_px:
            depth = self.focal_px * m
            rng = float(np.sqrt(depth ** 2 + off[0] ** 2 + off[1] ** 2))
        return Measurement(
            box=(x0, y0, x1, y1), disparity_px=disp, disparity_iqr_px=(float(q25), float(q75)),
            n_samples=int(len(d)), source=source, metres_per_px=m,
            rel_uncertainty=disparity_sigma_px / disp, offset_m=off,
            size_m=((x1 - x0) * m, (y1 - y0) * m) if box is not None else None,
            depth_m=depth, range_m=rng,
            sparse_disparity_px=float(np.median(d_sparse)) if len(d_sparse) else NAN, n_sparse=int(len(d_sparse)),
            dense_disparity_px=float(np.median(d_dense)) if d_dense.size else NAN, n_dense=int(d_dense.size),
        )


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _downscale(img: np.ndarray, max_res: int):
    h, w = img.shape[:2]
    if not max_res or max_res <= 0 or max(h, w) <= max_res:
        return img, 1.0, 1.0
    s = max_res / max(h, w)
    nw, nh = round(w * s), round(h * s)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), nw / w, nh / h


def match_features(gray_l: np.ndarray, gray_r: np.ndarray, ratio: float = 0.75):
    """SIFT keypoints + Lowe ratio test, one-to-one. Returns (N, 2) float64 pixel coords in each image."""
    sift = cv2.SIFT_create()
    kl, dl = sift.detectAndCompute(gray_l, None)
    kr, dr = sift.detectAndCompute(gray_r, None)
    if dl is None or dr is None or len(kl) < 2 or len(kr) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    best = {}
    for pair in matcher.knnMatch(dl, dr, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            m = pair[0]
            if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                best[m.trainIdx] = m
    good = list(best.values())
    pl = np.array([kl[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    pr = np.array([kr[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return pl, pr


def _dense_disparity(gl, gr, sparse_disp, mirrored, allow_negative=False, block_size=5):
    """SGBM raw shift for every left-image pixel (NaN = unknown), images assumed row-aligned.

    Raw shifts can be negative when one view of a 3D video was shifted (allow_negative).
    """
    if mirrored:  # second camera is on the left: mirror so the shift is positive, as SGBM expects
        gl, gr = np.ascontiguousarray(gl[:, ::-1]), np.ascontiguousarray(gr[:, ::-1])
    lo, hi = np.percentile(sparse_disp, [1, 99])         # search range from the matched points, with margin
    span = max(hi - lo, 1.0)
    min_d = int(np.floor(lo - 0.25 * span)) - 4
    if not allow_negative:
        min_d = max(0, min_d)
    num_d = int(np.clip(np.ceil((hi + 0.33 * span + 8 - min_d) / 16) * 16, 16, 16 * 48))
    sgbm = cv2.StereoSGBM_create(minDisparity=min_d, numDisparities=num_d, blockSize=block_size,
                                 P1=8 * block_size ** 2, P2=32 * block_size ** 2,
                                 disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    if mirrored:
        disp = disp[:, ::-1]
    disp[disp < min_d] = np.nan                           # SGBM marks failures with min_d - 1
    return disp


def stereo_from_pair(left: np.ndarray, right: np.ndarray, baseline_m: float,
                     focal_px: Optional[float] = None,
                     principal_point: Optional[Tuple[float, float]] = None,
                     dense: bool = True, max_res: int = 1600, ratio: float = 0.75,
                     max_row_diff_px: float = 2.0, disparity_offset_px: float = 0.0) -> StereoResult:
    """Match the two shots and measure how far every matched point shifts between them.

    left, right   : same-size images (BGR or grayscale) from the two cameras of the rig.
    baseline_m    : distance between the two cameras, metres.
    focal_px      : optional focal length in pixels at this resolution; only needed for distances.
    principal_point : optical centre (cx, cy), default image centre; only affects the offsets.
    dense         : also compute an SGBM disparity map aligned with `left`.
    max_res       : match on images downscaled to this longest side (speed); all
                    inputs/outputs stay in full-resolution pixels.
    max_row_diff_px : matches whose rows differ by more than this are rejected.
    disparity_offset_px : constant added to every shift. 0 for an ideal rig; non-zero when
                    the principal points differ (e.g. Middlebury's 'doffs') or the cameras toe in.
    """
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f'image sizes differ: {left.shape[:2]} vs {right.shape[:2]}')
    h, w = left.shape[:2]
    pp = tuple(principal_point) if principal_point is not None else (w / 2, h / 2)
    warnings: List[str] = []

    gl, sx, sy = _downscale(_to_gray(left), max_res)
    gr, _, _ = _downscale(_to_gray(right), max_res)
    pl, pr = match_features(gl, gr, ratio)
    n_matches = len(pl)
    if n_matches < 8:
        raise RuntimeError(f'only {n_matches} feature matches between the images; '
                           'they need to overlap and have some texture')
    scale = np.array([sx, sy])
    pl = (pl + 0.5) / scale - 0.5                         # back to full-resolution pixels
    pr = (pr + 0.5) / scale - 0.5

    dx, dy = pl[:, 0] - pr[:, 0], pl[:, 1] - pr[:, 1]
    on_row = np.abs(dy) <= max_row_diff_px
    if on_row.sum() < 8:
        raise RuntimeError(f'only {on_row.sum()} of {n_matches} matches lie on the same row in both '
                           'images - the shots are not horizontally aligned')
    if on_row.mean() < 0.5:
        warnings.append(f'only {on_row.mean():.0%} of matches lie on the same row in both images: the cameras '
                        'do not look aligned (vertical offset or rotation), measurements will be unreliable')
    # which camera is on the left: only guessed when no offset is given (an offset can make most
    # raw shifts negative even with the views in the right order, as in many 3D videos)
    sign = 1.0 if disparity_offset_px or np.median(dx[on_row]) >= 0 else -1.0
    if sign < 0:
        warnings.append('features move right from the left to the right image: assuming the "right" '
                        'camera is actually on the left')
    raw = sign * dx
    keep = on_row & (raw + disparity_offset_px > 0.5)
    if keep.sum() < 8:
        raise RuntimeError('too few matches with a usable shift - is the baseline too small for the scene?')

    res = StereoResult(baseline_m=baseline_m, image_size=(w, h), principal_point=pp,
                       pts_left=pl[keep], pts_right=pr[keep], disparity=raw[keep] + disparity_offset_px,
                       n_matches=n_matches, row_offset_px=float(np.median(dy[keep])),
                       focal_px=focal_px, warnings=warnings)
    if abs(res.row_offset_px) > 0.5:
        warnings.append(f'matches are on average {res.row_offset_px:+.2f} px apart vertically - '
                        'the cameras are not perfectly level with each other')

    if dense:
        disp_w = _dense_disparity(gl, gr, raw[keep] * sx, sign < 0, allow_negative=disparity_offset_px > 0)
        if disp_w.shape != (h, w):
            disp_w = cv2.resize(disp_w, (w, h), interpolation=cv2.INTER_NEAREST) / sx
        disp_full = disp_w + disparity_offset_px
        disp_full[~(disp_full > 0.5)] = np.nan
        res.disparity_map = disp_full
    return res


def measure(left, right, baseline_m, point=None, box=None, focal_px=None, **kwargs) -> Measurement:
    """One-shot helper: size / offset / (distance if focal_px is given) of the thing at `point` or in `box`."""
    return stereo_from_pair(left, right, baseline_m, focal_px, **kwargs).measure(point=point, box=box)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _disp_colors(d, lo, hi):
    """Turbo colours on a log scale: red = large shift = near, blue = far."""
    t = (np.log(np.clip(d, lo, hi)) - np.log(lo)) / max(np.log(hi) - np.log(lo), 1e-6)
    return cv2.applyColorMap((255 * t).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def disparity_range(res: StereoResult):
    return tuple(np.percentile(res.disparity, [2, 98]))


def draw_measurements(image: np.ndarray, res: StereoResult,
                      measurements: Sequence[Measurement] = (), max_points: int = 3000) -> np.ndarray:
    """Left image with matched points coloured by shift (red = near) and the measured regions labelled."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lo, hi = disparity_range(res)
    idx = np.random.default_rng(0).permutation(len(res.pts_left))[:max_points]
    colors = _disp_colors(res.disparity[idx], lo, hi)
    s = max(1, round(max(vis.shape[:2]) / 600))
    for (x, y), c in zip(res.pts_left[idx], colors):
        cv2.circle(vis, (int(round(x)), int(round(y))), 2 * s, tuple(int(v) for v in c), -1, cv2.LINE_AA)
    for m in measurements:
        x0, y0, x1, y1 = (int(round(v)) for v in m.box)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2 * s + 1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), s)
        parts = []
        if m.size_m is not None:
            parts.append(f'{m.size_m[0]:.2f}x{m.size_m[1]:.2f} m')
        if m.depth_m is not None:
            parts.append(f'{m.depth_m:.2f} m away')
        if not parts:
            parts.append(f'{1000 * m.metres_per_px:.2f} mm/px')
        label = ' | '.join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, s)
        ty = y0 - 6 if y0 - th - 10 > 0 else y1 + th + 6
        cv2.rectangle(vis, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), s, cv2.LINE_AA)
    return vis


def colorize_disparity(disparity_map: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.zeros(disparity_map.shape + (3,), np.uint8)
    ok = np.isfinite(disparity_map)
    out[ok] = _disp_colors(disparity_map[ok], lo, hi)
    return out

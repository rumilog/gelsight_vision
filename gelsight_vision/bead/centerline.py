"""Bead segmentation and centerline, computed in a plate-frame ortho image.

The D555 frame is resampled onto a regular plate-frame grid through the
camera model (`render_plate_ortho`), so every downstream quantity -- the
centerline, normals, visual width -- is directly in plate millimetres and
straight / curved / S-shaped beads are all handled the same way.

Segmentation is a white top-hat (bright structures narrower than
`max_bead_width_mm` survive, slow illumination gradients don't), Otsu
threshold, largest component, holes filled (the specular stripe along the
crown can read darker). Tuned on one rig snapshot (2026-11-04): expect to
revisit thresholds on new lighting / plates.

Caveat (measured): the bright region in the D555 image is ~2x wider than the
raised bead GelSight feels (15.5 vs 7-9 mm), and on that plate GelSight felt
the bead under one of the two bright stripes, ~3 mm off the bright-region
centre. Treat the centerline as a +/- few mm estimate until the spot-touch
calibration (calib_spot_touch) has been run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gelsight_vision.bead.camera_model import CameraModel


@dataclass
class PlateOrtho:
    image: np.ndarray        # (H, W, 3) float in [0, 1]
    valid: np.ndarray        # (H, W) bool, pixel landed inside the camera frame
    x0_mm: float
    y0_mm: float
    res_mm: float

    def ij_to_xy(self, ij) -> np.ndarray:
        """(col j, row i) ortho pixels -> plate (x, y) mm. Column = +x, row = +y."""
        ij = np.atleast_2d(np.asarray(ij, float))
        return np.c_[self.x0_mm + ij[:, 0] * self.res_mm, self.y0_mm + ij[:, 1] * self.res_mm]

    def xy_to_ij(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        return np.c_[(xy[:, 0] - self.x0_mm) / self.res_mm, (xy[:, 1] - self.y0_mm) / self.res_mm]


@dataclass
class BeadLine:
    xy_mm: np.ndarray        # (N, 2) centerline, uniformly spaced
    tangent: np.ndarray      # (N, 2) unit
    normal: np.ndarray       # (N, 2) unit, tangent rotated +90 deg
    s_mm: np.ndarray         # (N,) arc length
    width_mm: np.ndarray     # (N,) visual width normal to the line (nan where unknown)
    mask: np.ndarray         # ortho bead mask
    ortho: PlateOrtho

    @property
    def length_mm(self) -> float:
        return float(self.s_mm[-1]) if len(self.s_mm) else 0.0


def render_plate_ortho(image_rgb: np.ndarray, cam: CameraModel,
                       x_range_mm=(0.0, 150.0), y_range_mm=(0.0, 150.0),
                       res_mm: float = 0.5, z_surface_mm: float = 0.0) -> PlateOrtho:
    """Resample the camera image onto a plate-frame grid (bilinear)."""
    img = np.asarray(image_rgb, float)
    if img.max() > 1.5:
        img = img / 255.0
    nx = int(round((x_range_mm[1] - x_range_mm[0]) / res_mm))
    ny = int(round((y_range_mm[1] - y_range_mm[0]) / res_mm))
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    xs = x_range_mm[0] + jj.ravel() * res_mm
    ys = y_range_mm[0] + ii.ravel() * res_mm
    uv = cam.plate_to_pixel(np.c_[xs, ys, np.full(xs.size, z_surface_mm)])
    h, w = img.shape[:2]
    valid = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    chans = [ndi.map_coordinates(img[..., c], [uv[:, 1], uv[:, 0]], order=1, mode='nearest')
             for c in range(img.shape[2])]
    out = np.stack(chans, -1).reshape(ny, nx, -1)
    return PlateOrtho(out, valid.reshape(ny, nx), float(x_range_mm[0]), float(y_range_mm[0]),
                      float(res_mm))


def segment_bead(ortho: PlateOrtho, max_bead_width_mm: float = 25.0,
                 border_mm: float = 3.0, min_area_mm2: float = 100.0) -> np.ndarray:
    """Boolean bead mask in the ortho image (largest bright, narrow structure)."""
    res = ortho.res_mm
    v = ndi.gaussian_filter(ortho.image.mean(-1), 1.0)
    win = int(round(max_bead_width_mm / res)) | 1
    score = v - ndi.grey_opening(v, size=(win, win))
    b = max(1, int(border_mm / res))
    score[:b], score[-b:], score[:, :b], score[:, -b:] = 0, 0, 0, 0
    score[~ndi.binary_erosion(ortho.valid, iterations=b)] = 0
    hist, edges = np.histogram(score[score > 0], 256)
    if hist.sum() == 0:
        raise RuntimeError('segment_bead: empty score image')
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * edges[:-1])
    t = edges[np.nanargmax((mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12))]
    m = ndi.binary_closing(ndi.binary_opening(score > t, iterations=1), iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        raise RuntimeError('segment_bead: no bead found')
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes))
    if sizes[k] * res * res < min_area_mm2:
        raise RuntimeError(f'segment_bead: largest blob only {sizes[k] * res * res:.0f} mm^2')
    return ndi.binary_fill_holes(lab == (k + 1))


def _medial_path(mask: np.ndarray) -> np.ndarray:
    """Geodesic path between the two bead ends, weighted to hug the medial axis."""
    dt = ndi.distance_transform_edt(mask)
    ys, xs = np.nonzero(mask)
    idx = -np.ones(mask.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < mask.shape[0]) & (x2 >= 0) & (x2 < mask.shape[1])
        ok[ok] &= mask[y2[ok], x2[ok]]
        a, b = idx[ys[ok], xs[ok]], idx[y2[ok], x2[ok]]
        w = np.hypot(dy, dx) / (0.5 * (dt[ys[ok], xs[ok]] + dt[y2[ok], x2[ok]])) ** 2
        rows += [a, b]
        cols += [b, a]
        wts += [w, w]
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(ys), len(ys))).tocsr()
    d0 = dijkstra(G, indices=0)
    e1 = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1, pred = dijkstra(G, indices=e1, return_predecessors=True)
    e2 = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    path = [e2]
    while path[-1] != e1 and path[-1] >= 0:
        path.append(int(pred[path[-1]]))
    return np.c_[xs[path], ys[path]].astype(float)


def _widths(mask: np.ndarray, P: np.ndarray, N: np.ndarray, max_half_px: int) -> np.ndarray:
    t = np.arange(-max_half_px, max_half_px + 1, 0.5)
    c = len(t) // 2
    out = []
    for p, n in zip(P, N):
        q = p[None] + t[:, None] * n[None]
        v = ndi.map_coordinates(mask.astype(float), [q[:, 1], q[:, 0]], order=1) > 0.5
        if not v[c]:
            out.append(np.nan)
            continue
        lo = c
        while lo > 0 and v[lo - 1]:
            lo -= 1
        hi = c
        while hi < len(t) - 1 and v[hi + 1]:
            hi += 1
        out.append(t[hi] - t[lo] + 0.5)
    return np.asarray(out)


def bead_centerline(ortho: PlateOrtho, step_mm: float = 1.0, **seg_kwargs) -> BeadLine:
    """Detect the bead and return its centerline in plate mm."""
    mask = segment_bead(ortho, **seg_kwargs)
    pts = _medial_path(mask)
    if len(pts) < 8:
        raise RuntimeError('bead_centerline: centerline too short')
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 2.0)
    u = np.linspace(0, 1, 4000)
    x, y = splev(u, tck)
    L = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s_px = np.arange(0, L[-1], step_mm / ortho.res_mm)
    uu = np.interp(s_px, L, u)
    x, y = splev(uu, tck)
    dx, dy = splev(uu, tck, der=1)
    nrm = np.hypot(dx, dy)
    T = np.c_[dx / nrm, dy / nrm]
    N = np.c_[-T[:, 1], T[:, 0]]
    P = np.c_[x, y]
    width = _widths(mask, P, N, int(30 / ortho.res_mm)) * ortho.res_mm
    # ortho column = +x, row = +y with equal scale, so directions carry over unchanged
    return BeadLine(xy_mm=ortho.ij_to_xy(P), tangent=T, normal=N, s_mm=s_px * ortho.res_mm,
                    width_mm=width, mask=mask, ortho=ortho)

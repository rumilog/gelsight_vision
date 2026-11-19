"""Features inside one GelSight frame, from the raw frame vs the no-contact baseline.

Gel image coordinates (mm, origin at the frame centre):
    u -> right along the raw-image width  = the 19.9 mm long axis
    v -> down  along the raw-image height = the 11.2 mm short axis
How (u, v) map into the plate frame (sign / small rotation, and the gel
centre's offset from the gelsight_tooltip TCP) is what calib_spot_touch
measures; see planner.Calibration.gel_to_plate_dir.

The contact map is |frame - baseline| summed over channels. Regions where the
gel touches the surface change vs the baseline; regions out of contact don't
(they read near zero). On a 2.4-3.6 mm as-welded bead only a ~4 mm band along
the crown is in contact and both flanks read dark (measured 2026-11-16).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

GEL_FULL_WIDTH_MM = 19.9
GEL_FULL_HEIGHT_MM = 11.2


def _as_float(img) -> np.ndarray:
    a = np.asarray(img, float)
    return a if a.ndim == 3 else a[..., None]


def contact_map(raw, baseline, down: int = 4, sigma_px: float = 2.0) -> tuple[np.ndarray, float]:
    """Smoothed |raw - baseline| at 1/down resolution. Returns (map, mm_per_px)."""
    r, b = _as_float(raw), _as_float(baseline)
    if r.shape != b.shape:
        raise ValueError(f'frame {r.shape} and baseline {b.shape} differ')
    d = np.abs(r - b).sum(-1)
    if down > 1:
        h, w = (d.shape[0] // down) * down, (d.shape[1] // down) * down
        d = d[:h, :w].reshape(h // down, down, w // down, down).mean((1, 3))
    d = ndi.gaussian_filter(d, sigma_px)
    return d, GEL_FULL_WIDTH_MM / d.shape[1]


@dataclass
class CompactFeature:
    u_mm: float
    v_mm: float
    snr: float
    found: bool

    def to_dict(self) -> dict:
        return asdict(self)


def locate_compact_feature(raw, baseline, feature_mm: float = 1.5, border_mm: float = 0.8,
                           min_snr: float = 6.0) -> CompactFeature:
    """Strongest compact anomaly (a centre-punch dent, a small bump) in the frame.

    Difference-of-Gaussians on the contact map tuned to `feature_mm`, so a
    uniformly-contacting flat plate cancels out and a ~1-3 mm mark stands out.
    SNR = peak / robust noise (MAD) of the DoG response.
    """
    cmap, mmpx = contact_map(raw, baseline, sigma_px=0.0)
    s1 = max(0.5, 0.5 * feature_mm / mmpx / 2.0)
    dog = ndi.gaussian_filter(cmap, s1) - ndi.gaussian_filter(cmap, 4 * s1)
    resp = np.abs(dog)
    b = int(border_mm / mmpx)
    if b > 0:
        resp[:b], resp[-b:], resp[:, :b], resp[:, -b:] = 0, 0, 0, 0
    k = np.unravel_index(int(np.argmax(resp)), resp.shape)
    noise = 1.4826 * np.median(np.abs(resp - np.median(resp))) + 1e-9
    snr = float(resp[k] / noise)
    r = max(2, int(feature_mm / mmpx))
    y0, y1 = max(0, k[0] - r), min(resp.shape[0], k[0] + r + 1)
    x0, x1 = max(0, k[1] - r), min(resp.shape[1], k[1] + r + 1)
    win = resp[y0:y1, x0:x1]
    win = np.clip(win - 0.5 * resp[k], 0, None)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy, cx = (yy * win).sum() / win.sum(), (xx * win).sum() / win.sum()
    h, w = resp.shape
    return CompactFeature(u_mm=float((cx - (w - 1) / 2) * mmpx),
                          v_mm=float((cy - (h - 1) / 2) * mmpx), snr=snr, found=snr >= min_snr)


@dataclass
class CrownReading:
    crown_u_mm: float          # crown position along the long axis (0 = frame centre)
    band_width_mm: float       # width of the in-contact band around the crown
    contact_fraction: float    # share of the frame in contact
    edge_contact_left: float   # mean contact level in the first 15% of u / band level
    edge_contact_right: float
    plate_both_sides: bool     # heuristic: gel touches something at both ends of u
    crown_near_edge: bool      # crown within 2 mm of a frame end -> re-touch

    def to_dict(self) -> dict:
        return asdict(self)


def read_crown(raw, baseline, edge_frac: float = 0.15, contact_rel: float = 0.25,
               edge_rel: float = 0.35) -> CrownReading:
    """Locate the bead crown along u for a touch with the long axis ACROSS the bead.

    Profile = median over v of the contact map (a bead crossing the frame shows
    as a band of contact perpendicular to u). The crown is the profile peak
    (parabolic sub-pixel refine). Thresholds are first guesses from the
    2026-11-16 frames; recalibrate them from calib_bead_sweep output.
    """
    cmap, mmpx = contact_map(raw, baseline)
    prof = ndi.gaussian_filter1d(np.median(cmap, axis=0), 2.0)
    base = float(np.percentile(prof, 5))
    p = prof - base
    i = int(np.argmax(p))
    if 0 < i < len(p) - 1:
        den = p[i - 1] - 2 * p[i] + p[i + 1]
        di = 0.5 * (p[i - 1] - p[i + 1]) / den if den != 0 else 0.0
    else:
        di = 0.0
    n = len(p)
    peak = max(float(p[i]), 1e-9)
    above = p > contact_rel * peak
    lab, _ = ndi.label(above)
    band = lab == lab[i]
    e = max(1, int(edge_frac * n))
    left, right = float(p[:e].mean() / peak), float(p[-e:].mean() / peak)
    crown_u = float((i + di - (n - 1) / 2) * mmpx)
    half = GEL_FULL_WIDTH_MM / 2
    c0 = np.percentile(cmap, 5)
    return CrownReading(
        crown_u_mm=crown_u,
        band_width_mm=float(band.sum() * mmpx),
        contact_fraction=float((cmap - c0 > contact_rel * (cmap.max() - c0)).mean()),
        edge_contact_left=left, edge_contact_right=right,
        plate_both_sides=bool(left > edge_rel and right > edge_rel),
        crown_near_edge=bool(abs(crown_u) > half - 2.0))

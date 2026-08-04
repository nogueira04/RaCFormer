"""C2 redesign, image-side variant EDT: edge DISTANCE TRANSFORM instead of gradient magnitude.

The original C2 score sampled a blurred Sobel MAGNITUDE at the projected radar pixels. That
score is monotone in "points land on strong image response" and has no notion of distance to
a specific edge, so the optimiser drifts the point set onto whatever region of the image has
the most texture (foliage, building facades) and the clean extrinsic is not a local maximum.

Levinson and Thrun (RSS 2013, doi 10.15607/RSS.2013.IX.029) solve exactly this for
lidar-camera by replacing the raw edge image with an INVERSE DISTANCE TRANSFORM,

    D_ij = alpha * E_ij + (1 - alpha) * max_xy ( E_xy * gamma^{max(|x-i|,|y-j|)} )

with alpha = 1/3 and gamma = 0.98 (their Eq. 1). D decays monotonically with the Chebyshev
distance to the nearest strong edge, so a point that lands ON an edge is a strict local
maximum of D in the image plane -- the property the gradient-magnitude score lacked. Here D
is computed as alpha * Ehat + (1 - alpha) * gamma^dist with dist the exact DIST_C distance
transform of a thresholded edge mask, which is the same field up to the per-edge strength
weighting and is far cheaper. Levinson and Thrun note (their footnote 2) that the choice of
edge filter is not important.

The second mechanism taken from that paper is the DISCRIMINABILITY STATISTIC F_C: the
fraction of grid-neighbour perturbations of a candidate calibration C whose objective is
WORSE than at C. F_C is a rank, so it is invariant to the absolute amount of texture in the
scene, which is precisely what defeated the original score. Levinson and Thrun report
F_C = 99.7% (sd 1.4) for correct calibrations against 50.5% (sd 14) for incorrect ones over
a 9-frame window of a 64-beam lidar. ``discriminability`` below computes the same statistic
on a whole nuScenes scene (about 40 keyframes) so that the two populations can be compared
directly on this sensor suite.

If F_C at the clean extrinsic is not clearly separated from F_C at a perturbed extrinsic,
then no optimiser over any monotone function of this objective can recover the extrinsic,
and the image channel is dead for this purpose regardless of the search strategy.
"""

import numpy as np

import c2_recalib as c2

ALPHA = 1.0 / 3.0
GAMMA = 0.98
EDGE_PERCENTILE = 95.0     # pixels above this percentile of |grad| are "edges"


class EdtSceneCache(c2.SceneCameraCache):
    """SceneCameraCache with the gradient pyramid replaced by a single-level inverse
    distance transform of the edge map. All projection geometry is inherited unchanged."""

    def __init__(self, nusc, scene_token, cfg=None, cameras=c2.CAMERA_CHANNELS,
                 alpha=ALPHA, gamma=GAMMA, edge_pct=EDGE_PERCENTILE):
        import cv2

        cfg = cfg or c2.ScoreConfig(levels=((1.0, 0.01),))
        super(EdtSceneCache, self).__init__(nusc, scene_token, cfg=cfg, cameras=cameras)
        G = self.G[0]                                  # (S, C, H, W) uint8 |grad|
        out = np.zeros_like(G)
        S, C = G.shape[0], G.shape[1]
        for s in range(S):
            for c in range(C):
                g = G[s, c]
                thr = np.percentile(g, edge_pct)
                mask = (g >= max(1.0, thr)).astype(np.uint8)
                dist = cv2.distanceTransform(1 - mask, cv2.DIST_C, 3)
                d = alpha * (g.astype(np.float32) / 255.0) + (1.0 - alpha) * (gamma ** dist)
                out[s, c] = np.clip(d * 255.0, 0, 255).astype(np.uint8)
        self.G = [np.ascontiguousarray(out)]
        self.level_shapes = [self.level_shapes[0]]
        self.alpha, self.gamma, self.edge_pct = alpha, gamma, edge_pct


def discriminability(sensor_cache, radius=0.02, level=0):
    """Levinson and Thrun F_C, restricted to the three ROTATION axes.

    Returns the fraction of the 26 grid-neighbour rotation perturbations (each component in
    {-radius, 0, +radius}, excluding the centre) whose objective is WORSE than at the
    extrinsic currently set on the cache. A correctly calibrated sensor should sit near 1.0
    and an incorrectly calibrated one near 0.5.
    """
    zero = np.zeros(3)
    j0, _ = sensor_cache.score(zero, zero, level)
    worse = 0
    total = 0
    for a in (-radius, 0.0, radius):
        for b in (-radius, 0.0, radius):
            for c in (-radius, 0.0, radius):
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                total += 1
                j, _ = sensor_cache.score(np.array([a, b, c]), zero, level)
                if j < j0:
                    worse += 1
    return float(worse) / float(total), j0

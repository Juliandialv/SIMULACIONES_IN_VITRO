"""Processing configuration parameters for the SIV viewer."""

# ── PointCloud processing defaults ───────────────────────────────────────────
DEFAULT_VOXEL_SIZE  = 1.5
DEFAULT_DOWNSAMPLE_RATIO = 0.25

# ── Cranial plane analysis ───────────────────────────────────────────────────
EAR_CLEARANCE_RATIO = 0.25
CONTOUR_BAND_MM = 2.0
N_CRANIAL_LEVELS = 10

# ── Surface reconstruction ───────────────────────────────────────────────────
POISSON_DEPTH = 6
DENSITY_QUANTILE = 0.1

"""Unified defect taxonomy for automotive / paint CMPL (RGB-only track)."""

# Cross-task scratch family — single source of truth (also re-exported by
# common.project_paths.SCRATCH_CLASS_NAMES for backward compatibility).
#   cardd.scratch + carparts.Scratch + syndcar.sd_scratch  -> is_scratch=1
SCRATCH_CLASS_NAMES = frozenset({
    'scratch',   # cardd
    'Scratch',   # carparts
    'sd_scratch',  # syndcar
})

# Phase 0 multi-class (CarDD COCO category names -> folder)
CARDD_COCO_TO_FOLDER = {
    'dent': 'dent',
    'scratch': 'scratch',
    'crack': 'crack',
    'glass shatter': 'glass_shatter',
    'lamp broken': 'lamp_broken',
    'tire flat': 'tire_flat',
}

CARDD_PRETRAIN_CLASSES = (
    'scratch', 'dent', 'crack', 'glass_shatter',
)

# car_parts_damage 8 个原生类 (CMPL v6 新增)
CARPARTS_PRETRAIN_CLASSES = (
    'Scratch',          # 划痕 (与 cardd.scratch 同语义, scratch_aux=1)
    'Dent',             # 凹坑 (与 cardd.dent 同语义)
    'Cracked',          # 裂纹 (与 cardd.crack 同语义)
    'Paint chip',       # 漆面崩裂
    'Flaking',          # 漆面剥落
    'Broken part',      # 部件缺失/断裂
    'Missing part',     # 部件缺失
    'Corrosion',        # 腐蚀/锈蚀
)

# Phase 0 NEU-DET (300 images per class, balanced)
NEU_FILENAME_PREFIX_TO_FOLDER = {
    'crazing': 'crazing',
    'inclusion': 'inclusion',
    'patches': 'patches',
    'pitted': 'pitted_surface',
    'rolled-in': 'rolled_in_scale',
    'scratches': 'scratches',
}

NEU_PRETRAIN_CLASSES = tuple(NEU_FILENAME_PREFIX_TO_FOLDER.values())

# Meta-learning 2-way folders (compatible with AviationDefectDataset class scan)
CMPL_CLASS_FOLDERS = {
    0: '0_scratch',
    1: '3_dent',
}

CMPL_META_TWO_WAY = ('0_scratch', '3_dent')

# COCO category id -> meta folder (2-way: scratch vs dent only)
CARDD_COCO_ID_TO_META_FOLDER = {
    1: '3_dent',
    2: '0_scratch',
}

DEFAULT_CROP_POLICY = {
    'expand_ratio': 1.8,
    'min_defect_ratio': 0.15,
    'min_short_side': 384,
    'max_short_side': 640,
}

# Track 2 (recommended): CarDD train/val -> Phase0 only; ASDE -> meta-train; CarDD test -> holdout
CMPL_DOMAIN_SPLIT = {
    'meta_train_domains': (0,),
    'meta_test_domain': 3,
    'cardd_split_to_domain': {
        'train': (),       # Phase0 pretrain only
        'val': (),         # Phase0 pretrain only
        'test': (3,),      # meta-test holdout
    },
    'asde_meta_domain': 0,
}

# Legacy: CarDD train/val in meta-train (domain 0+2) — not recommended
CMPL_DOMAIN_SPLIT_LEGACY = {
    'meta_train_domains': (0, 2),
    'meta_test_domain': 3,
    'cardd_split_to_domain': {
        'train': (0,),
        'val': (2,),
        'test': (3,),
    },
}

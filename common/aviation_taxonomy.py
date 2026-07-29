"""Unified defect taxonomy shared by conversion scripts."""

UNIFIED_CLASS = {
    'scratch': 0,
    'dirty': 1,
    'stain': 1,
    'dot': 1,
    'crease': 2,
    'damage': 3,
}

CLASS_FOLDERS = {
    0: '0_scratch',
    1: '1_contamination',
    2: '2_crease',
    3: '3_damage',
}

NORMAL_FOLDER = '0_normal'

AEBAD_DEFECT_TO_CLASS = {
    'ablation': 3,
    'breakdown': 3,
    'fracture': 3,
    'groove': 0,
}

AEBAD_DEFECT_TO_DOMAIN = {
    'ablation': 4,
    'breakdown': 5,
    'fracture': 6,
    'groove': 7,
}

BLADESYNTH_CLASSES = ('Normal', 'Scratch', 'Nick', 'Dent', 'Corrosion')
BLADESYNTH_DEFECT_CLASSES = ('Scratch', 'Nick', 'Dent', 'Corrosion')

DEFAULT_CROP_POLICY = {
    'expand_ratio': 1.8,
    'min_defect_ratio': 0.15,
    'min_short_side': 384,
    'max_short_side': 640,
}

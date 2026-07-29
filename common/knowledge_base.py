"""Material / defect priors for aviation multimodal fusion."""

import json
import os

# ρ, λ, E, Cp, T_melt, hardness, roughness, thickness, porosity, fiber_vol, resin_vol, damping
DEFAULT_MATERIAL_PARAMS = [0.0] * 12

KNOWLEDGE_BASE = {
    'AL7075': {
        'params': [2.81, 130.0, 72.0, 960.0, 628.0, 150.0, 1.6, 3.0, 0.01, 0.0, 0.0, 0.02],
        'defect_weights': {
            'surface_burr': [0.80, 0.05, 0.15],
            'delamination': [0.10, 0.25, 0.65],
            'thermal_damage': [0.20, 0.70, 0.10],
            'hole_offset': [0.70, 0.15, 0.15],
        },
    },
    'TC4': {
        'params': [4.51, 7.4, 103.0, 520.0, 1933.0, 350.0, 1.2, 4.0, 0.005, 0.0, 0.0, 0.01],
        'defect_weights': {
            'thermal_damage': [0.10, 0.85, 0.05],
            'surface_burr': [0.75, 0.15, 0.10],
        },
    },
    'CFRP': {
        'params': [1.60, 0.1, 70.0, 710.0, 350.0, 200.0, 2.0, 2.5, 0.02, 0.60, 0.40, 0.05],
        'defect_weights': {
            'fiber_breakage': [0.50, 0.00, 0.50],
            'delamination': [0.05, 0.00, 0.95],
            'hole_offset': [0.65, 0.05, 0.30],
        },
    },
}

DEFAULT_FUSION_PRIOR = [0.45, 0.30, 0.25]


def load_domain_meta(data_dir, domain_id, dataset_name='aviation'):
    """Load optional domain metadata JSON for TaskEncoder inputs."""
    path = os.path.join(data_dir, dataset_name, 'domain_{}'.format(domain_id), 'meta.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def lookup_material(material_id):
    entry = KNOWLEDGE_BASE.get(material_id, {})
    params = entry.get('params', DEFAULT_MATERIAL_PARAMS)
    return {
        'material_id': material_id,
        'params': params,
        'defect_weights': entry.get('defect_weights', {}),
    }


def fusion_prior_tensor(material_id, defect_type=None):
    import torch

    entry = KNOWLEDGE_BASE.get(material_id, {})
    weights = entry.get('defect_weights', {})
    if defect_type and defect_type in weights:
        prior = weights[defect_type]
    else:
        prior = DEFAULT_FUSION_PRIOR
    return torch.tensor(prior, dtype=torch.float32)

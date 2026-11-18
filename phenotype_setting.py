"""Phenotype categories, defaults, and controlled variants."""


from dataclasses import dataclass


LEVELS = {"small": 0.75, "medium": 1.0, "large": 1.3}


ANTENNA_LEVELS = {"short": 0.45, "medium": 0.65, "long": 0.90}


LEG_LEVELS = {"short": 0.75, "medium": 1.0, "long": 1.25}


EYE_LEVELS = {
    "tiny": .35, "extra_small": .55, "small": .75, "medium": .95,
    "large": 1.15, "extra_large": 1.35, "huge": 1.55, "giant": 1.75,
}


MANDIBLE_LEVELS = {"small": 0.75, "medium": 1.30, "large": 2.00}


BRISTLE_LEVELS = {"none": 0, "sparse": 14, "medium": 36, "dense": 64}


BRISTLE_LENGTH_LEVELS = {"short": .65, "medium": 1.0, "long": 1.4}


GAUSSIAN_SCALE_STD = .01


@dataclass(frozen=True)
class AntPhenotype:
    head_scale: float = 1.0
    eye_scale: float = 1.0
    antenna_scale: float = 1.0
    mandible_scale: float = 1.0
    thorax_scale: float = 1.0
    abdomen_scale: float = 1.0
    leg_scale: float = 1.0
    waist_scale: float = 1.0
    bristle_count: int = 18
    bristle_scale: float = 1.0
    seed: int = 0


def make_phenotype_variant(trait: str, level: str) -> AntPhenotype:
    """Change one categorical trait while all other traits stay medium."""
    values = {
        "head_scale": LEVELS["medium"],
        "thorax_scale": LEVELS["medium"],
        "abdomen_scale": LEVELS["medium"],
        "waist_scale": LEVELS["medium"],
        "eye_scale": EYE_LEVELS["medium"],
        "antenna_scale": ANTENNA_LEVELS["medium"],
        "mandible_scale": MANDIBLE_LEVELS["medium"],
        "leg_scale": LEG_LEVELS["medium"],
        "bristle_count": BRISTLE_LEVELS["sparse"],
        "bristle_scale": BRISTLE_LENGTH_LEVELS["medium"],
        "seed": 0,
    }
    scale_fields = {
        "head": ("head_scale", LEVELS),
        "thorax": ("thorax_scale", LEVELS),
        "abdomen": ("abdomen_scale", LEVELS),
        "waist": ("waist_scale", LEVELS),
        "eye": ("eye_scale", EYE_LEVELS),
        "antenna": ("antenna_scale", ANTENNA_LEVELS),
        "mandible": ("mandible_scale", MANDIBLE_LEVELS),
        "leg": ("leg_scale", LEG_LEVELS),
        "bristle_length": ("bristle_scale", BRISTLE_LENGTH_LEVELS),
    }
    if trait == "bristle_density":
        values["bristle_count"] = BRISTLE_LEVELS[level]
    else:
        field, levels = scale_fields[trait]
        values[field] = levels[level]
    return AntPhenotype(**values)

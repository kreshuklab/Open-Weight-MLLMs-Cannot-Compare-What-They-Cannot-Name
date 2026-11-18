import argparse
import itertools
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from phenotype_setting import (
    ANTENNA_LEVELS,
    BRISTLE_LENGTH_LEVELS,
    BRISTLE_LEVELS,
    EYE_LEVELS,
    GAUSSIAN_SCALE_STD,
    LEG_LEVELS,
    LEVELS,
    MANDIBLE_LEVELS,
    AntPhenotype,
    make_phenotype_variant
)
from draw_ant import (
    HIGHLIGHT_GROUPS,
    common_foreground_box,
    highlight_anatomy,
    render_ant,
    save_phenotype_preview
)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument(
        "--num-pairs", type=int, default=100,
        help=("Number of random pairs; images may recur across pairs. "
              "Default: pair each image once. Use 0 to skip pairing."),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty --output-dir dataset",
    )
    ap.add_argument("--size", type=int, default=1024,help="size must be at least 128")
    ap.add_argument(
        "--random-colored-phenotype", action="store_true",
        help="Highlight exactly one random group: legs, antenna, or mandible",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--phenotype-preview",
        type=Path,
        nargs="?",
        const=Path("ant_reference_phenotype_preview.png"),
        help=(
            "Generate a controlled phenotype comparison sheet. Optionally "
            "provide its output path."),)
    return ap



def dataset_sample_id(index: int) -> str:
    """Identify each ant by its unpadded number."""
    return f"ant_{index}"


def generate_random_pairs(
    root: Path, records: list[dict], seed: int, num_pairs: int | None = None,
) -> None:
    """Save random pairs, using disjoint pairs when no count is specified."""
    if num_pairs is not None and num_pairs < 0:
        raise ValueError("--num-pairs must be non-negative")
    if num_pairs and len(records) < 2:
        raise ValueError("At least two images are required to generate pairs")
    pair_dir = root / "random_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    shuffled = records.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    count = len(shuffled) // 2 if num_pairs is None else num_pairs
    with (pair_dir / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(count):
            left, right = (
                shuffled[2 * index:2 * index + 2]
                if num_pairs is None else rng.sample(shuffled, 2)
            )
            pair_id = f"pair_{index + 1:05d}"
            relative = Path("random_pairs") / f"{pair_id}.png"
            with Image.open(root / left["image"]) as left_image, Image.open(
                root / right["image"]
            ) as right_image:
                canvas = Image.new(
                    "RGB",
                    (left_image.width + right_image.width,
                     max(left_image.height, right_image.height)),
                    "white",
                )
                canvas.paste(left_image.convert("RGB"), (0, 0))
                canvas.paste(right_image.convert("RGB"), (left_image.width, 0))
                canvas.save(root / relative, optimize=True)
            stream.write(json.dumps({
                "pair_id": pair_id,
                "image": relative.as_posix(),
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_image": left["image"],
                "right_image": right["image"],
            }) + "\n")
    print(f"Generated {count} pairs: {pair_dir}")
    if num_pairs is None and len(shuffled) % 2:
        print(f"Unpaired sample: {shuffled[-1]['sample_id']}")


def generate_dataset(args: argparse.Namespace) -> None:
    """Generate reproducibly shuffled phenotype combinations and metadata."""
    if args.num_images < 0:
        raise ValueError("--num-images must be non-negative")
    if args.num_pairs is not None and args.num_pairs < 0:
        raise ValueError("--num-pairs must be non-negative")
    if args.num_pairs and args.num_images < 2:
        raise ValueError("At least two images are required to generate pairs")
    root = args.output_dir.resolve()
    images = root / "images"
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {root}. Use --overwrite to replace it."
            )
        shutil.rmtree(root)
    images.mkdir(parents=True, exist_ok=True)
    traits = [
        ("head", LEVELS), ("thorax", LEVELS),
        ("abdomen", LEVELS), ("waist", LEVELS),
        ("eye", EYE_LEVELS), ("antenna", ANTENNA_LEVELS),
        ("mandible", MANDIBLE_LEVELS), ("leg", LEG_LEVELS),
        ("bristle_density", BRISTLE_LEVELS),
        ("bristle_length", BRISTLE_LENGTH_LEVELS),
    ]
    combinations = list(itertools.product(*(list(levels) for _name, levels in traits)))
    rng = random.Random(args.seed)
    selected = []
    while len(selected) < args.num_images:
        cycle = combinations.copy()
        rng.shuffle(cycle)
        selected.extend(cycle)

    records = []
    for index, values in enumerate(selected[:args.num_images]):
        category = dict(zip((name for name, _levels in traits), values))
        varied = lambda table, key: table[category[key]]*rng.gauss(1, GAUSSIAN_SCALE_STD)
        scales = {
            "head": varied(LEVELS, "head"),
            "thorax": varied(LEVELS, "thorax"),
            "abdomen": varied(LEVELS, "abdomen"),
            "waist": varied(LEVELS, "waist"),
            "eye": varied(EYE_LEVELS, "eye"),
            "antenna": varied(ANTENNA_LEVELS, "antenna"),
            "mandible": varied(MANDIBLE_LEVELS, "mandible"),
            "leg": varied(LEG_LEVELS, "leg"),
            "bristle": varied(BRISTLE_LENGTH_LEVELS, "bristle_length"),
        }
        count = BRISTLE_LEVELS[category["bristle_density"]]
        phenotype = AntPhenotype(
            head_scale=scales["head"], eye_scale=scales["eye"],
            antenna_scale=scales["antenna"], mandible_scale=scales["mandible"],
            thorax_scale=scales["thorax"], abdomen_scale=scales["abdomen"],
            leg_scale=scales["leg"], waist_scale=scales["waist"], bristle_count=count,
            bristle_scale=scales["bristle"], seed=args.seed+index,
        )
        highlighted = (
            rng.choice(list(HIGHLIGHT_GROUPS))
            if args.random_colored_phenotype else None
        )
        sample_id = dataset_sample_id(index)
        relative = Path("images") / f"{sample_id}.png"
        image = render_ant(phenotype, args.size)
        if highlighted is not None:
            image = highlight_anatomy(image, highlighted)
        image.save(root / relative, optimize=True)
        records.append({"sample_id": sample_id, "image": relative.as_posix(),
                        **category, **{f"{key}_scale": round(value, 6)
                        for key, value in scales.items()}, "bristle_count": count,
                        "highlighted_anatomy": highlighted,
                        "highlight_color": "green" if highlighted else None,
                        "seed": args.seed+index})
        if (index+1) % 10 == 0:
            print(f"Generated {index+1}/{args.num_images}")
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record)+"\n")
    generate_random_pairs(root, records, args.seed, args.num_pairs)
    print(images)
    print(root / "metadata.jsonl")


def main() -> int:
    args = parser().parse_args()
    if args.phenotype_preview is not None:
        print(save_phenotype_preview(args.phenotype_preview, args.size))
        return 0
    if args.output_dir is not None:
        generate_dataset(args)
        return 0
    phenotype = AntPhenotype(
        head_scale=LEVELS["medium"], eye_scale=EYE_LEVELS["medium"],
        antenna_scale=ANTENNA_LEVELS["medium"],
        mandible_scale=MANDIBLE_LEVELS["medium"],
        thorax_scale=LEVELS["medium"], abdomen_scale=LEVELS["medium"],
        leg_scale=LEG_LEVELS["medium"], waist_scale=LEVELS["medium"],
        bristle_count=BRISTLE_LEVELS["sparse"],
        bristle_scale=BRISTLE_LENGTH_LEVELS["medium"], seed=args.seed,
    )
    image = render_ant(phenotype, args.size)
    if args.random_colored_phenotype:
        highlighted = random.Random(args.seed).choice(list(HIGHLIGHT_GROUPS))
        image = highlight_anatomy(image, highlighted)
    output = Path("parametric_ant.png")
    image.save(output, optimize=True)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    main()

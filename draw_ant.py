"""Render ant images and apply anatomy highlighting."""


import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from phenotype_setting import (
    ANTENNA_LEVELS,
    BRISTLE_LENGTH_LEVELS,
    LEG_LEVELS,
    MANDIBLE_LEVELS,
    AntPhenotype,
    make_phenotype_variant,
)


GRAY = {
    "abdomen": 62,
    "waist": 96,
    "thorax": 118,
    "head": 142,
    "legs": 158,
    "mandibles": 158,
    "antennae": 158,
    "eyes": 238,
    "bristles": 250,
}


HIGHLIGHT_GROUPS = {
    "legs": GRAY["legs"],
    "antenna": GRAY["antennae"],
    "mandible": GRAY["mandibles"],
}


def _ellipse_box(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int]:
    return tuple(round(v) for v in (cx-w/2, cy-h/2, cx+w/2, cy+h/2))


def _draw_segment_bristles(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    count: int,
    length: int,
    stroke: int,
    rng: random.Random,
) -> None:
    """Attach evenly spaced setae perpendicular to a polyline."""
    if count <= 0 or len(points) < 2:
        return
    segments = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        segment_length = math.hypot(end[0]-start[0], end[1]-start[1])
        segments.append((start, end, segment_length))
        total += segment_length
    if total <= 0:
        return

    for index in range(count):
        distance = total*(index+1)/(count+1)
        traversed = 0.0
        for start, end, segment_length in segments:
            if distance <= traversed+segment_length:
                t = (distance-traversed)/max(segment_length, 1e-6)
                x = start[0]+(end[0]-start[0])*t
                y = start[1]+(end[1]-start[1])*t
                tx = (end[0]-start[0])/max(segment_length, 1e-6)
                ty = (end[1]-start[1])/max(segment_length, 1e-6)
                direction = -1 if index % 2 == 0 else 1
                local_length = length*rng.uniform(.80, 1.20)
                tip = (
                    round(x-direction*ty*local_length),
                    round(y+direction*tx*local_length),
                )
                draw.line([(round(x), round(y)), tip], fill=GRAY["bristles"], width=stroke)
                break
            traversed += segment_length


def render_ant(p: AntPhenotype, size: int = 1024) -> Image.Image:
    """Render one exclusive-class mask in the reference generator's style."""
    work = 4
    s = size*work
    k = s/1024.0
    pt = lambda x, y: (round(x*k), round(y*k))
    width = lambda value: max(1, round(value*k))
    canvas = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(canvas)

    # Centers follow a fixed top-view body axis; dimensions are phenotype driven.
    head = (512, 245, 145*p.head_scale, 170*p.head_scale)
    thorax = (512, 455, 132*p.thorax_scale, 205*p.thorax_scale)
    waist = (512, 625, 55*p.waist_scale, 55*p.waist_scale)
    abdomen = (512, 790, 205*p.abdomen_scale, 310*p.abdomen_scale)

    # Bristles originate on ellipse boundaries and point along the outward
    # surface normal. They are drawn before the body so their roots are joined
    # cleanly to the corresponding segment.
    rng = random.Random(p.seed)
    bristle_parts = [head, thorax, abdomen]
    if p.bristle_count:
        base_allocations = [
            p.bristle_count // 3,
            p.bristle_count // 4,
            p.bristle_count - p.bristle_count // 3 - p.bristle_count // 4,
        ]
        allocations = [max(1, round(count*1.30)) for count in base_allocations]
        for (cx, cy, bw, bh), count in zip(bristle_parts, allocations):
            phase = rng.uniform(0.0, 2.0*math.pi)
            for index in range(count):
                theta = phase + 2.0*math.pi*index/max(count, 1)
                rx, ry = bw/2.0, bh/2.0
                x0, y0 = cx+rx*math.cos(theta), cy+ry*math.sin(theta)
                nx = math.cos(theta)/max(rx, 1e-6)
                ny = math.sin(theta)/max(ry, 1e-6)
                norm = math.hypot(nx, ny)
                nx, ny = nx/norm, ny/norm
                length = rng.uniform(11.0, 22.0)*p.bristle_scale
                draw.line(
                    [pt(x0, y0), pt(x0+nx*length, y0+ny*length)],
                    fill=GRAY["bristles"],
                    width=width(3),
                )

    # Legs are behind the body. Each side has three articulated legs.
    # Roots sit inside the thorax silhouette so all legs remain visibly joined
    # after the body is drawn over their attachment points.
    leg_origins = [(488, 395), (486, 485), (500, 535)]
    leg_shapes = [
        [(0, 0), (-125, -55), (-220, -145), (-290, -205)],
        [(0, 0), (-145, 25), (-225, 120), (-320, 170)],
        [(0, 0), (-105, 105), (-145, 225), (-215, 305)],
    ]
    leg_w = width(14)
    for side in (-1, 1):
        for (ox, oy), offsets in zip(leg_origins, leg_shapes):
            base_x = 512 + side*(512-ox)
            points = [
                pt(base_x+side*abs(dx)*p.leg_scale, oy+dy*p.leg_scale)
                for dx, dy in offsets
            ]
            draw.line(points, fill=GRAY["legs"], width=leg_w, joint="curve")
            _draw_segment_bristles(
                draw, points, p.bristle_count//4, width(13), width(2), rng
            )

    # Connect the head and thorax even when their phenotype sizes are small.
    head_bottom = head[1] + head[3] / 2.0
    thorax_top = thorax[1] - thorax[3] / 2.0
    neck_top, neck_bottom = sorted((head_bottom, thorax_top))
    neck_half_width = max(8.0, 12.0*min(p.head_scale, p.thorax_scale))
    draw.rectangle(
        (
            round((512.0-neck_half_width)*k), round(neck_top*k),
            round((512.0+neck_half_width)*k), round(neck_bottom*k),
        ),
        fill=GRAY["thorax"],
    )

    # Continuous central bridge prevents background gaps between the thorax,
    # two petiole nodes, and abdomen. The nodes are drawn over this bridge so
    # their individual shapes remain visible.
    thorax_bottom = thorax[1] + thorax[3] / 2.0
    abdomen_top = abdomen[1] - abdomen[3] / 2.0
    bridge_top, bridge_bottom = sorted((thorax_bottom, abdomen_top))
    bridge_width = max(12.0, 24.0 * p.waist_scale)
    draw.rectangle(
        (
            round((512.0 - bridge_width / 2.0) * k),
            round(bridge_top * k),
            round((512.0 + bridge_width / 2.0) * k),
            round(bridge_bottom * k),
        ),
        fill=GRAY["waist"],
    )

    # Body sections, ordered back-to-front at their narrow connections.
    for (cx, cy, w, h), class_name in (
        (abdomen, "abdomen"), (waist, "waist"),
        (thorax, "thorax"), (head, "head"),
    ):
        draw.ellipse(_ellipse_box(cx*k, cy*k, w*k, h*k), fill=GRAY[class_name])

    # Antennae: symmetric elbowed polylines controlled independently of head size.
    for side in (-1, 1):
        root = (512+side*62*p.head_scale, 184)
        points = [
            # Begin inside the head to keep the antenna visibly attached.
            pt(512+side*48*p.head_scale, 205),
            pt(*root),
            pt(root[0]+side*62*p.antenna_scale, root[1]-74*p.antenna_scale),
            pt(root[0]+side*112*p.antenna_scale, root[1]-108*p.antenna_scale),
        ]
        draw.line(points, fill=GRAY["antennae"], width=width(12), joint="curve")
        _draw_segment_bristles(
            draw, points, p.bristle_count//2, width(12), width(2), rng
        )

    # Small second petiole maintains the characteristic ant waist.
    draw.ellipse(
        _ellipse_box(512*k, 590*k, 48*p.waist_scale*k, 40*p.waist_scale*k),
        fill=GRAY["waist"],
    )

    # Eyes are dark ovals and remain independently scalable.
    eye_w, eye_h = 18*p.eye_scale*k, 31*p.eye_scale*k
    for side in (-1, 1):
        draw.ellipse(
            _ellipse_box((512+side*55*p.head_scale)*k, 245*k, eye_w, eye_h),
            fill=GRAY["eyes"],
        )

    # Paired curved mandibles above the head, approximated by outlined hooks.
    jaw_w = width(11)
    # Keep the medium mandible's shape and opening angle fixed. Both control
    # points are scaled by the same scalar from the root, so phenotype changes
    # affect length only (not direction, curvature proportions, or width).
    jaw_length_ratio = p.mandible_scale / MANDIBLE_LEVELS["medium"]
    medium_bend_vector = (15.208, -45.6)
    medium_tip_vector = (16.32, -75.72)
    for side in (-1, 1):
        # Anchor the root inside the phenotype-scaled head. Since mandibles are
        # drawn after the head, keep a short visible overlap to guarantee a
        # continuous attachment for every head/mandible size combination.
        jaw_root_x = 512+side*min(30.0, 45.0*p.head_scale)
        jaw_root_y = 245-70.0*p.head_scale
        points = [
            pt(jaw_root_x, jaw_root_y),
            pt(
                jaw_root_x + side*medium_bend_vector[0]*jaw_length_ratio,
                jaw_root_y + medium_bend_vector[1]*jaw_length_ratio,
            ),
            pt(
                jaw_root_x + side*medium_tip_vector[0]*jaw_length_ratio,
                jaw_root_y + medium_tip_vector[1]*jaw_length_ratio,
            ),
        ]
        draw.line(points, fill=GRAY["mandibles"], width=jaw_w, joint="curve")
        _draw_segment_bristles(
            draw, points, p.bristle_count//4, width(9), width(2), rng
        )

    return canvas.resize((size, size), Image.Resampling.NEAREST)


def highlight_anatomy(image: Image.Image, group: str) -> Image.Image:
    """Highlight exactly one grayscale semantic group in green."""
    luminance = np.asarray(image.convert("L"), dtype=np.uint8)
    output = np.repeat(luminance[:, :, None], 3, axis=2)
    selected = luminance == HIGHLIGHT_GROUPS[group]
    output[selected, 0] = 0
    output[selected, 1] = luminance[selected]
    output[selected, 2] = 0
    return Image.fromarray(output, mode="RGB")


def common_foreground_box(images: list[Image.Image]) -> tuple[int, int, int, int]:
    """Return one union foreground box that can be applied to every image."""
    boxes = [box for image in images if (box := image.getbbox()) is not None]
    if not boxes:
        return (0, 0, images[0].width, images[0].height)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )

def save_phenotype_preview(path: Path, size: int = 512) -> Path:
    """Save a labeled sheet and each controlled phenotype as a separate image."""
    phenotype_levels = {
        "antenna": ANTENNA_LEVELS,
        "mandible": MANDIBLE_LEVELS,
        "leg": LEG_LEVELS,
        "bristle_length": BRISTLE_LENGTH_LEVELS,
    }
    phenotype_labels = {
        "antenna": "antenna length",
        "mandible": "mandible length",
        "leg": "leg length",
        "bristle_length": "bristle length",
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = path.parent / f"{path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rendered = {
        (trait, level): render_ant(make_phenotype_variant(trait, level), size)
        for trait, levels in phenotype_levels.items()
        for level in levels
    }
    crop_box = common_foreground_box(list(rendered.values()))

    font = ImageFont.load_default()
    tile = 250
    header_height = 27
    label_height = 23
    row_height = tile + header_height + label_height
    columns = max(len(levels) for levels in phenotype_levels.values())
    sheet = Image.new("L", (columns * tile, len(phenotype_levels) * row_height), 220)
    draw = ImageDraw.Draw(sheet)

    for row, (trait, levels) in enumerate(phenotype_levels.items()):
        y = row * row_height
        draw.text(
            (8, y + 7),
            f"Controlled phenotype: {phenotype_labels[trait]}",
            fill=15,
            font=font,
        )
        for column, level in enumerate(levels):
            # Apply exactly the same crop to every phenotype. No image is
            # resized, so their relative anatomical sizes remain comparable.
            preview = rendered[(trait, level)].crop(crop_box)
            phenotype_name = phenotype_labels[trait].replace(" ", "_")
            preview.save(
                image_dir / f"{phenotype_name}_{level}.png",
                optimize=True,
            )
            thumb_size = tile - 20
            x = column * tile
            sheet.paste(
                preview.resize((thumb_size, thumb_size), Image.Resampling.NEAREST),
                (x + 10, y + header_height),
            )
            draw.text(
                (x + 12, y + header_height + thumb_size + 5),
                level.replace("_", " "),
                fill=35,
                font=font,
            )

    sheet.save(path, optimize=True)
    return path

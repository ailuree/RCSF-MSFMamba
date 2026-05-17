from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class MethodFigure:
    title: str
    run_name: str


@dataclass(frozen=True)
class ZoomRegion:
    name: str
    box: tuple[int, int, int, int]


HOUSTON2013_METHODS = (
    MethodFigure("GT", "full_model_h2013"),
    MethodFigure("Baseline", "baseline_h2013"),
    MethodFigure("Reliability Gate", "gate_h2013"),
    MethodFigure("Cross-State", "cross_h2013"),
    MethodFigure("Robust Training", "robust_h2013"),
    MethodFigure("Full Model v1", "full_model_h2013"),
)

HOUSTON2018_SMALL_METHODS = (
    MethodFigure("GT", "full_model_h2018_small"),
    MethodFigure("Baseline", "baseline_h2018_small"),
    MethodFigure("Reliability Gate", "gate_h2018_small"),
    MethodFigure("Cross-State", "cross_h2018_small"),
    MethodFigure("Robust Training", "robust_h2018_small"),
    MethodFigure("Full Model v1", "full_model_h2018_small"),
)

ZOOM_METHODS = (
    MethodFigure("GT", "full_model_h2013"),
    MethodFigure("Baseline", "baseline_h2013"),
    MethodFigure("Robust Training", "robust_h2013"),
    MethodFigure("Full Model v1", "full_model_h2013"),
)

HOUSTON2013_ZOOM_REGIONS = (
    ZoomRegion("Region A", (160, 20, 420, 220)),
    ZoomRegion("Region B", (1060, 15, 1380, 235)),
)

ERROR_METHODS = (
    MethodFigure("Baseline Error", "baseline_h2013"),
    MethodFigure("Robust Error", "robust_h2013"),
    MethodFigure("Full Model Error", "full_model_h2013"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 5 paper-ready figures from checkpoint visualization outputs."
    )
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("./checkpoints"),
        help="Root directory that contains dataset experiment folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./checkpoints/paper_figures/chapter5"),
        help="Directory used to save generated paper figures.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidate_paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def ensure_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image file: {path}")
    return Image.open(path).convert("RGB")


def get_figure_path(checkpoints_root: Path, dataset: str, run_name: str, image_name: str) -> Path:
    return checkpoints_root / dataset / run_name / "figures" / image_name


def add_canvas_border(image: Image.Image, color: tuple[int, int, int] = (90, 90, 90), width: int = 2) -> Image.Image:
    canvas = Image.new("RGB", (image.width + width * 2, image.height + width * 2), "white")
    canvas.paste(image, (width, width))
    drawer = ImageDraw.Draw(canvas)
    drawer.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=color, width=width)
    return canvas


def add_title_strip(image: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    title_height = 42
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    drawer = ImageDraw.Draw(canvas)
    drawer.rectangle((0, 0, image.width, title_height), fill=(245, 245, 245))
    bbox = drawer.textbbox((0, 0), title, font=font)
    text_x = (image.width - (bbox[2] - bbox[0])) // 2
    text_y = (title_height - (bbox[3] - bbox[1])) // 2 - 1
    drawer.text((text_x, text_y), title, fill="black", font=font)
    return canvas


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.NEAREST)


def compose_grid(
    items: list[Image.Image],
    columns: int,
    cell_width: int,
    gutter: int,
    margin: int,
    background: str = "white",
) -> Image.Image:
    rows = (len(items) + columns - 1) // columns
    resized_items = [resize_to_width(image, cell_width) for image in items]
    cell_heights = [0] * rows
    for index, image in enumerate(resized_items):
        row = index // columns
        cell_heights[row] = max(cell_heights[row], image.height)

    canvas_width = margin * 2 + columns * cell_width + (columns - 1) * gutter
    canvas_height = margin * 2 + sum(cell_heights) + (rows - 1) * gutter
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)

    y = margin
    for row in range(rows):
        x = margin
        for col in range(columns):
            index = row * columns + col
            if index >= len(resized_items):
                break
            image = resized_items[index]
            offset_x = x + (cell_width - image.width) // 2
            offset_y = y + (cell_heights[row] - image.height) // 2
            canvas.paste(image, (offset_x, offset_y))
            x += cell_width + gutter
        y += cell_heights[row] + gutter

    return canvas


def build_input_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    run_name = "full_model_h2013"
    images = [
        add_title_strip(
            ensure_image(get_figure_path(checkpoints_root, "Houston2013", run_name, image_name)),
            title,
            font,
        )
        for title, image_name in (
            ("HSI RGB", "hsi_rgb.png"),
            ("Auxiliary View", "aux_view.png"),
            ("Ground Truth", "gt_map.png"),
        )
    ]
    figure = compose_grid(images, columns=3, cell_width=560, gutter=28, margin=28)
    output_path = output_dir / "fig5_1_houston2013_inputs.png"
    figure.save(output_path)
    return output_path


def build_overview_figure(
    checkpoints_root: Path,
    dataset: str,
    methods: tuple[MethodFigure, ...],
    output_path: Path,
    font: ImageFont.ImageFont,
) -> Path:
    images: list[Image.Image] = []
    for method in methods:
        image_name = "gt_map.png" if method.title == "GT" else "prediction_map.png"
        image = ensure_image(get_figure_path(checkpoints_root, dataset, method.run_name, image_name))
        images.append(add_title_strip(image, method.title, font))

    figure = compose_grid(images, columns=3, cell_width=560, gutter=28, margin=28)
    figure.save(output_path)
    return output_path


def crop_with_border(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    crop = image.crop(box)
    return add_canvas_border(crop, width=2)


def build_roi_overview_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    gt_image = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "gt_map.png"))
    drawer = ImageDraw.Draw(gt_image)
    for region in HOUSTON2013_ZOOM_REGIONS:
        drawer.rectangle(region.box, outline=(255, 255, 255), width=4)
        drawer.rectangle(region.box, outline=(220, 20, 60), width=2)
        label_x = region.box[0] + 8
        label_y = max(6, region.box[1] + 8)
        label_bbox = drawer.textbbox((0, 0), region.name, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        drawer.rectangle(
            (label_x - 4, label_y - 2, label_x + label_w + 4, label_y + label_h + 2),
            fill=(255, 255, 255),
        )
        drawer.text((label_x, label_y), region.name, fill=(220, 20, 60), font=font)

    titled = add_title_strip(add_canvas_border(gt_image), "Houston2013 ROI Overview", font)
    output_path = output_dir / "fig5_roi_houston2013_overview.png"
    titled.save(output_path)
    return output_path


def build_error_map(prediction: Image.Image, ground_truth: Image.Image) -> Image.Image:
    if prediction.size != ground_truth.size:
        raise ValueError("Prediction map and ground-truth map must have identical sizes for error-map generation.")
    pred = prediction.convert("RGB")
    gt = ground_truth.convert("RGB")
    pred_pixels = pred.load()
    gt_pixels = gt.load()
    error = Image.new("RGB", pred.size, (0, 0, 0))
    error_pixels = error.load()
    for y in range(pred.height):
        for x in range(pred.width):
            gt_color = gt_pixels[x, y]
            if gt_color == (0, 0, 0):
                error_pixels[x, y] = (0, 0, 0)
            elif pred_pixels[x, y] != gt_color:
                error_pixels[x, y] = (220, 20, 60)
            else:
                error_pixels[x, y] = (25, 25, 25)
    return error


def build_error_map_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    gt_image = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "gt_map.png"))
    images: list[Image.Image] = [
        add_title_strip(add_canvas_border(gt_image), "Ground Truth", font),
    ]
    for method in ERROR_METHODS:
        prediction = ensure_image(get_figure_path(checkpoints_root, "Houston2013", method.run_name, "prediction_map.png"))
        error_map = build_error_map(prediction, gt_image)
        images.append(add_title_strip(add_canvas_border(error_map), method.title, font))

    figure = compose_grid(images, columns=2, cell_width=820, gutter=28, margin=28)
    output_path = output_dir / "fig5_error_houston2013.png"
    figure.save(output_path)
    return output_path


def build_zoom_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    source_images = {}
    for method in ZOOM_METHODS:
        image_name = "gt_map.png" if method.title == "GT" else "prediction_map.png"
        source_images[method.title] = ensure_image(
            get_figure_path(checkpoints_root, "Houston2013", method.run_name, image_name)
        )

    rows: list[Image.Image] = []
    for region in HOUSTON2013_ZOOM_REGIONS:
        row_items: list[Image.Image] = []
        for method in ZOOM_METHODS:
            crop = crop_with_border(source_images[method.title], region.box)
            title = f"{region.name} - {method.title}"
            row_items.append(add_title_strip(crop, title, font))
        row_image = compose_grid(row_items, columns=4, cell_width=390, gutter=24, margin=20)
        rows.append(row_image)

    figure = compose_grid(rows, columns=1, cell_width=1648, gutter=24, margin=0)
    output_path = output_dir / "fig5_3_houston2013_zoom.png"
    figure.save(output_path)
    return output_path


def save_manifest(output_dir: Path, paths: dict[str, Path]) -> None:
    manifest_path = output_dir / "chapter5_manifest.json"
    serializable = {key: str(value.resolve()) for key, value in paths.items()}
    serializable["houston2013_zoom_regions"] = [
        {"name": region.name, "box": list(region.box)} for region in HOUSTON2013_ZOOM_REGIONS
    ]
    manifest_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoints_root = args.checkpoints_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    title_font = load_font(size=22)

    outputs = {
        "fig5_1": build_input_figure(checkpoints_root, output_dir, title_font),
        "fig5_2": build_overview_figure(
            checkpoints_root=checkpoints_root,
            dataset="Houston2013",
            methods=HOUSTON2013_METHODS,
            output_path=output_dir / "fig5_2_houston2013_overview.png",
            font=title_font,
        ),
        "fig5_3": build_zoom_figure(checkpoints_root, output_dir, title_font),
        "fig5_4": build_overview_figure(
            checkpoints_root=checkpoints_root,
            dataset="Houston2018",
            methods=HOUSTON2018_SMALL_METHODS,
            output_path=output_dir / "fig5_4_houston2018_small_overview.png",
            font=title_font,
        ),
        "roi_overview": build_roi_overview_figure(checkpoints_root, output_dir, title_font),
        "error_map": build_error_map_figure(checkpoints_root, output_dir, title_font),
    }
    save_manifest(output_dir, outputs)

    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

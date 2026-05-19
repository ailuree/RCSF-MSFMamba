from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np


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

HOUSTON2013_IMPROVEMENT_REGIONS = (
    ZoomRegion("Region A", (1520, 70, 1800, 290)),
    ZoomRegion("Region B", (960, 0, 1240, 220)),
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


def build_input_figure_vertical(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
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
    figure = compose_grid(images, columns=1, cell_width=900, gutter=20, margin=24)
    output_path = output_dir / "fig5_1_houston2013_inputs_vertical.png"
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


def build_improvement_map(
    baseline: Image.Image,
    target: Image.Image,
    ground_truth: Image.Image,
) -> Image.Image:
    if baseline.size != target.size or baseline.size != ground_truth.size:
        raise ValueError("All maps must have identical sizes for improvement-map generation.")
    base = baseline.convert("RGB")
    pred = target.convert("RGB")
    gt = ground_truth.convert("RGB")
    canvas = Image.new("RGB", base.size, (0, 0, 0))
    canvas_pixels = canvas.load()
    base_pixels = base.load()
    pred_pixels = pred.load()
    gt_pixels = gt.load()
    for y in range(base.height):
        for x in range(base.width):
            gt_color = gt_pixels[x, y]
            if gt_color == (0, 0, 0):
                canvas_pixels[x, y] = (0, 0, 0)
                continue
            base_ok = base_pixels[x, y] == gt_color
            pred_ok = pred_pixels[x, y] == gt_color
            if (not base_ok) and pred_ok:
                canvas_pixels[x, y] = (0, 200, 80)   # corrected by improved method
            elif base_ok and (not pred_ok):
                canvas_pixels[x, y] = (220, 20, 60)  # regression
            elif pred_ok:
                canvas_pixels[x, y] = (35, 35, 35)
            else:
                canvas_pixels[x, y] = (100, 100, 100)
    return canvas


def compute_improvement_stats(
    baseline: Image.Image,
    target: Image.Image,
    ground_truth: Image.Image,
    box: tuple[int, int, int, int] | None = None,
) -> dict[str, float | int]:
    if baseline.size != target.size or baseline.size != ground_truth.size:
        raise ValueError("All maps must have identical sizes for improvement-stat calculation.")
    base = baseline.convert("RGB")
    pred = target.convert("RGB")
    gt = ground_truth.convert("RGB")
    base_pixels = base.load()
    pred_pixels = pred.load()
    gt_pixels = gt.load()

    if box is None:
        x_start, y_start, x_end, y_end = 0, 0, base.width, base.height
    else:
        x_start, y_start, x_end, y_end = box

    corrected = 0
    regressed = 0
    both_right = 0
    both_wrong = 0
    labeled = 0
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            gt_color = gt_pixels[x, y]
            if gt_color == (0, 0, 0):
                continue
            labeled += 1
            base_ok = base_pixels[x, y] == gt_color
            pred_ok = pred_pixels[x, y] == gt_color
            if (not base_ok) and pred_ok:
                corrected += 1
            elif base_ok and (not pred_ok):
                regressed += 1
            elif pred_ok:
                both_right += 1
            else:
                both_wrong += 1

    return {
        "labeled": labeled,
        "corrected": corrected,
        "regressed": regressed,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "net_gain": corrected - regressed,
        "corrected_ratio": corrected / labeled if labeled else 0.0,
        "regressed_ratio": regressed / labeled if labeled else 0.0,
    }


def overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / min(area_a, area_b)


def auto_select_improvement_regions(
    baseline: Image.Image,
    target: Image.Image,
    ground_truth: Image.Image,
    region_width: int = 280,
    region_height: int = 220,
    stride_x: int = 80,
    stride_y: int = 70,
    top_k: int = 2,
    min_corrected: int = 15,
    min_net_gain: int = 12,
) -> list[ZoomRegion]:
    candidates: list[tuple[dict[str, float | int], tuple[int, int, int, int]]] = []
    width = baseline.width
    height = baseline.height
    for y in range(0, max(1, height - region_height + 1), stride_y):
        for x in range(0, max(1, width - region_width + 1), stride_x):
            box = (x, y, x + region_width, y + region_height)
            stats = compute_improvement_stats(baseline, target, ground_truth, box)
        if stats["labeled"] < max(200, region_width * region_height * 0.05):
            continue
        if stats["corrected"] < min_corrected or stats["net_gain"] < min_net_gain:
            continue
        candidates.append((stats, box))

    candidates.sort(
        key=lambda item: (
            item[0]["net_gain"],
            item[0]["corrected"],
            -item[0]["regressed"],
            item[0]["corrected_ratio"],
        ),
        reverse=True,
    )

    selected: list[ZoomRegion] = []
    for stats, box in candidates:
        if stats["net_gain"] < min_net_gain:
            break
        if any(overlap_ratio(box, region.box) > 0.35 for region in selected):
            continue
        selected.append(ZoomRegion(f"Region {chr(ord('A') + len(selected))}", box))
        if len(selected) >= top_k:
            break

    return selected


def load_metrics_summary(checkpoints_root: Path, dataset: str, run_name: str, filename: str) -> dict[str, float]:
    path = checkpoints_root / dataset / run_name / "metrics" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_per_class_accuracy(checkpoints_root: Path, dataset: str, run_name: str, filename: str = "test_per_class_accuracy.csv") -> list[tuple[str, float]]:
    path = checkpoints_root / dataset / run_name / "metrics" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-class accuracy file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(row["class_name"], float(row["accuracy"])) for row in reader]


def build_improvement_focus_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    gt_image = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "gt_map.png"))
    baseline = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "baseline_h2013", "prediction_map.png"))
    full = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "prediction_map.png"))
    regions = list(HOUSTON2013_IMPROVEMENT_REGIONS)

    panels: list[Image.Image] = []
    for region in regions:
        stats = compute_improvement_stats(baseline, full, gt_image, region.box)
        region_gt = add_title_strip(crop_with_border(gt_image, region.box), f"{region.name} - GT", font)
        region_base = add_title_strip(crop_with_border(baseline, region.box), f"{region.name} - Baseline", font)
        region_full = add_title_strip(crop_with_border(full, region.box), f"{region.name} - Full Model v1", font)

        improve_map = build_improvement_map(baseline, full, gt_image)
        region_improve = add_title_strip(
            crop_with_border(improve_map, region.box),
            f"{region.name} - Delta +{stats['corrected']}/-{stats['regressed']}",
            font,
        )
        row = compose_grid(
            [region_gt, region_base, region_full, region_improve],
            columns=4,
            cell_width=390,
            gutter=24,
            margin=20,
        )
        panels.append(row)

    legend = Image.new("RGB", (1648, 92), "white")
    drawer = ImageDraw.Draw(legend)
    drawer.text((20, 16), "Green: corrected pixels", fill=(0, 120, 40), font=font)
    drawer.text((440, 16), "Red: regressed pixels", fill=(180, 20, 60), font=font)
    drawer.text((810, 16), "Gray/Black: unchanged pixels", fill=(60, 60, 60), font=font)
    drawer.text((20, 54), "ROIs are selected from positive net-gain regions to show where the full model visibly corrects baseline errors.", fill=(70, 70, 70), font=font)
    panels.append(legend)

    figure = compose_grid(panels, columns=1, cell_width=1648, gutter=18, margin=0)
    output_path = output_dir / "fig5_3_houston2013_improvement_focus.png"
    figure.save(output_path)
    return output_path


def build_method_advantage_figure(checkpoints_root: Path, output_dir: Path, font: ImageFont.ImageFont) -> Path:
    gt_image = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "gt_map.png"))
    baseline = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "baseline_h2013", "prediction_map.png"))
    robust = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "robust_h2013", "prediction_map.png"))
    full = ensure_image(get_figure_path(checkpoints_root, "Houston2013", "full_model_h2013", "prediction_map.png"))

    images: list[Image.Image] = [
        add_title_strip(add_canvas_border(gt_image), "Ground Truth", font),
        add_title_strip(
            add_canvas_border(build_improvement_map(baseline, robust, gt_image)),
            "Robust vs Baseline",
            font,
        ),
        add_title_strip(
            add_canvas_border(build_improvement_map(baseline, full, gt_image)),
            "Full Model v1 vs Baseline",
            font,
        ),
    ]
    main_panel = compose_grid(images, columns=1, cell_width=1180, gutter=18, margin=18)

    legend_font = load_font(size=18)
    legend = Image.new("RGB", (main_panel.width, 64), "white")
    drawer = ImageDraw.Draw(legend)
    items = [
        ((0, 200, 80), "Corrected"),
        ((220, 20, 60), "Regressed"),
        ((100, 100, 100), "Both wrong"),
        ((35, 35, 35), "Both correct"),
    ]
    x = 24
    y = 18
    for color, text in items:
        drawer.rectangle((x, y, x + 28, y + 20), fill=color, outline=(80, 80, 80))
        drawer.text((x + 40, y - 1), text, fill="black", font=legend_font)
        x += 240

    figure = compose_grid([main_panel, legend], columns=1, cell_width=main_panel.width, gutter=8, margin=0)
    output_path = output_dir / "fig5_4_houston2013_method_advantage.png"
    figure.save(output_path)
    return output_path


def build_metrics_dashboard_figure(checkpoints_root: Path, output_dir: Path) -> Path:
    methods = [
        ("Baseline", "baseline_h2013", "#9aa3ad"),
        ("Reliability Gate", "gate_h2013", "#5b8ff9"),
        ("Cross-State", "cross_h2013", "#f6bd16"),
        ("Robust Training", "robust_h2013", "#5ad8a6"),
        ("Full Model v1", "full_model_h2013", "#e8684a"),
    ]
    metrics = ["oa", "aa", "kappa", "macro_f1"]
    metric_titles = ["OA", "AA", "Kappa", "Macro-F1"]
    values = {metric: [] for metric in metrics}
    labels = []
    colors = []
    for label, run_name, color in methods:
        summary = load_metrics_summary(checkpoints_root, "Houston2013", run_name, "test_metrics_summary.json")
        labels.append(label)
        colors.append(color)
        for metric in metrics:
            values[metric].append(float(summary[metric]))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = values[metric]
        bars = ax.barh(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        min_val = min(vals)
        max_val = max(vals)
        pad = max(0.3, (max_val - min_val) * 0.2)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_title(metric_titles[idx], fontsize=13, pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10)
        if idx % 2 == 1:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=10)
    fig.suptitle("Houston2013 Overall Metric Comparison", fontsize=16)
    output_path = output_dir / "fig5_extra_houston2013_metrics_dashboard.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_per_class_gain_figure(checkpoints_root: Path, output_dir: Path) -> Path:
    baseline = dict(load_per_class_accuracy(checkpoints_root, "Houston2013", "baseline_h2013"))
    full = dict(load_per_class_accuracy(checkpoints_root, "Houston2013", "full_model_h2013"))
    class_names = list(baseline.keys())
    full_gain = np.array([full[name] - baseline[name] for name in class_names], dtype=float)
    order = np.argsort(np.abs(full_gain))[::-1][:8]
    ordered_names = [class_names[idx] for idx in order]
    ordered_full = full_gain[order]

    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    y = np.arange(len(ordered_names))
    colors = ["#e8684a" if value >= 0 else "#4c78a8" for value in ordered_full]
    ax.barh(y, ordered_full, height=0.52, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_xlabel("Per-class Accuracy Gain (%)", fontsize=11)
    ax.set_title("Houston2013 Class-wise Gain of Full Model v1 over Baseline", fontsize=15, pad=10)
    for yy, value in zip(y, ordered_full):
        ax.text(value + (0.12 if value >= 0 else -0.12), yy, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.text(0.99, 0.03, "Red: improved   Blue: declined", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#444444")
    output_path = output_dir / "fig5_extra_houston2013_per_class_gain.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_robustness_bar_figure(checkpoints_root: Path, output_dir: Path) -> Path:
    modes = [
        ("Normal", "test_metrics_summary.json"),
        ("Drop HSI", "test_drop_hsi_metrics_summary.json"),
        ("Drop Aux", "test_drop_aux_metrics_summary.json"),
        ("Noise", "test_noise_metrics_summary.json"),
        ("Combined", "test_combined_metrics_summary.json"),
    ]
    robust_vals = []
    full_vals = []
    for _, filename in modes:
        robust_vals.append(float(load_metrics_summary(checkpoints_root, "Houston2013", "robust_h2013", filename)["oa"]))
        full_vals.append(float(load_metrics_summary(checkpoints_root, "Houston2013", "full_model_h2013", filename)["oa"]))

    x = np.arange(len(modes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.6, 6.2), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, robust_vals, width=width, color="#5ad8a6", edgecolor="black", linewidth=0.6, label="Robust Training")
    bars2 = ax.bar(x + width / 2, full_vals, width=width, color="#e8684a", edgecolor="black", linewidth=0.6, label="Full Model v1")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in modes], fontsize=10)
    ax.set_ylabel("OA (%)", fontsize=11)
    ax.set_title("Houston2013 Robustness Comparison under Degradation", fontsize=15, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ymin = min(min(robust_vals), min(full_vals)) - 2.5
    ymax = max(max(robust_vals), max(full_vals)) + 2.0
    ax.set_ylim(ymin, ymax)
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    output_path = output_dir / "fig5_extra_houston2013_robustness.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
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
        "fig5_1_vertical": build_input_figure_vertical(checkpoints_root, output_dir, title_font),
        "fig5_2": build_overview_figure(
            checkpoints_root=checkpoints_root,
            dataset="Houston2013",
            methods=HOUSTON2013_METHODS,
            output_path=output_dir / "fig5_2_houston2013_overview.png",
            font=title_font,
        ),
        "fig5_3": build_zoom_figure(checkpoints_root, output_dir, title_font),
        "fig5_3_improvement": build_improvement_focus_figure(checkpoints_root, output_dir, title_font),
        "fig5_4": build_overview_figure(
            checkpoints_root=checkpoints_root,
            dataset="Houston2018",
            methods=HOUSTON2018_SMALL_METHODS,
            output_path=output_dir / "fig5_4_houston2018_small_overview.png",
            font=title_font,
        ),
        "roi_overview": build_roi_overview_figure(checkpoints_root, output_dir, title_font),
        "error_map": build_error_map_figure(checkpoints_root, output_dir, title_font),
        "method_advantage": build_method_advantage_figure(checkpoints_root, output_dir, title_font),
        "metrics_dashboard": build_metrics_dashboard_figure(checkpoints_root, output_dir),
        "per_class_gain": build_per_class_gain_figure(checkpoints_root, output_dir),
        "robustness_bar": build_robustness_bar_figure(checkpoints_root, output_dir),
    }
    save_manifest(output_dir, outputs)

    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
import os
import time
from datetime import datetime
from operator import truediv

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.io import loadmat, savemat
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix


DATASET_CLASS_NAMES = {
    "Berlin": [
        "Forest",
        "Residential Area",
        "Industrial Area",
        "Low Plants",
        "Soil",
        "Allotment",
        "Commercial Area",
        "Water",
    ],
    "Augsburg": [
        "Forest",
        "Residential Area",
        "Industrial Area",
        "Low Plants",
        "Allotment",
        "Commercial Area",
        "Water",
    ],
    "Houston2013": [
        "Healthy grass",
        "Stressed grass",
        "Synthetic grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking Lot 1",
        "Parking Lot 2",
        "Tennis Court",
        "Running Track",
    ],
    "Houston2018": [
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ],
}


DATASET_COLOR_MAPS = {
    "Houston2013": [
        (0, 0, 0),
        (20, 153, 76),
        (92, 184, 92),
        (145, 214, 118),
        (17, 109, 63),
        (184, 134, 11),
        (40, 119, 237),
        (218, 112, 214),
        (199, 21, 133),
        (219, 68, 55),
        (255, 140, 0),
        (255, 215, 0),
        (46, 139, 87),
        (70, 130, 180),
        (138, 43, 226),
        (255, 99, 71),
    ],
    "Houston2018": [
        (0, 0, 0),
        (20, 153, 76),
        (92, 184, 92),
        (144, 238, 144),
        (0, 100, 0),
        (34, 139, 34),
        (184, 134, 11),
        (40, 119, 237),
        (176, 196, 222),
        (128, 0, 128),
        (220, 20, 60),
        (255, 140, 0),
        (255, 215, 0),
        (255, 99, 71),
        (205, 92, 92),
        (139, 69, 19),
        (255, 182, 193),
        (255, 228, 181),
        (0, 191, 255),
        (105, 105, 105),
        (255, 250, 205),
    ],
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_dataset_info(dataset, config_path="dataset_info.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data[dataset]


def get_dataset_class_names(dataset_name):
    class_names = DATASET_CLASS_NAMES.get(dataset_name)
    if class_names is None:
        raise ValueError(f"Unsupported dataset for reporting: {dataset_name}")
    return class_names


def get_dataset_palette(dataset_name):
    if dataset_name not in DATASET_COLOR_MAPS:
        raise ValueError(f"Unsupported dataset for visualization: {dataset_name}")
    return DATASET_COLOR_MAPS[dataset_name]


def create_run_structure(save_root, dataset, run_name=""):
    dataset_root = ensure_dir(os.path.join(save_root, dataset))
    final_run_name = run_name or timestamp_string()
    run_dir = ensure_dir(os.path.join(dataset_root, final_run_name))
    paths = {
        "dataset_root": dataset_root,
        "run_dir": run_dir,
        "weights_dir": ensure_dir(os.path.join(run_dir, "weights")),
        "logs_dir": ensure_dir(os.path.join(run_dir, "logs")),
        "metrics_dir": ensure_dir(os.path.join(run_dir, "metrics")),
        "figures_dir": ensure_dir(os.path.join(run_dir, "figures")),
        "predictions_dir": ensure_dir(os.path.join(run_dir, "predictions")),
    }
    return paths


def get_latest_run_dir(save_root, dataset):
    dataset_root = os.path.join(save_root, dataset)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"No run directory found under {dataset_root}")
    subdirs = [
        os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No experiment runs found under {dataset_root}")
    return max(subdirs, key=os.path.getmtime)


def resolve_run_dir(save_root, dataset, run_name=""):
    if run_name:
        run_dir = os.path.join(save_root, dataset, run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir
    return get_latest_run_dir(save_root, dataset)


def resolve_checkpoint(save_root, dataset, checkpoint_path="", run_name="", prefer="best"):
    if checkpoint_path:
        final_checkpoint = checkpoint_path
        if not os.path.isabs(final_checkpoint):
            final_checkpoint = os.path.abspath(final_checkpoint)
        if not os.path.exists(final_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {final_checkpoint}")
        run_dir = os.path.dirname(os.path.dirname(final_checkpoint))
        return run_dir, final_checkpoint

    run_dir = resolve_run_dir(save_root, dataset, run_name)
    weights_dir = os.path.join(run_dir, "weights")
    candidate_names = ["best.pth", "last.pth"] if prefer == "best" else ["last.pth", "best.pth"]
    for name in candidate_names:
        checkpoint = os.path.join(weights_dir, name)
        if os.path.exists(checkpoint):
            return run_dir, checkpoint
    raise FileNotFoundError(f"No checkpoint found under {weights_dir}")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def save_csv(rows, headers, path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def append_history_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def AA_andEachClassAccuracy(confusion):
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def evaluate_model(net, data_loader, dataset_name, device, print_freq=20):
    class_names = get_dataset_class_names(dataset_name)
    labels = list(range(len(class_names)))
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None

    net.eval()
    start_time = time.time()
    pred_list = []
    true_list = []
    coord_list = []
    sample_count = 0

    with torch.inference_mode():
        for batch_idx, (_, x, hsi_pca, test_labels, h, w) in enumerate(data_loader, start=1):
            hsi_pca = hsi_pca.to(device)
            x = x.to(device)
            _, outputs = net(hsi_pca.unsqueeze(1), x)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels_np = test_labels.detach().cpu().numpy()
            coords_np = np.stack((h.detach().cpu().numpy(), w.detach().cpu().numpy()), axis=1)

            pred_list.append(preds)
            true_list.append(labels_np)
            coord_list.append(coords_np)
            sample_count += preds.shape[0]

            if total_batches is not None and (
                batch_idx == 1 or batch_idx % print_freq == 0 or batch_idx == total_batches
            ):
                elapsed = time.time() - start_time
                print(
                    f"Eval Step [{batch_idx:04d}/{total_batches:04d}] "
                    f"Samples: {sample_count} "
                    f"Elapsed: {elapsed:.2f}s"
                )

            del hsi_pca, x, outputs, preds, labels_np, coords_np

    y_pred = np.concatenate(pred_list, axis=0) if pred_list else np.array([], dtype=np.int64)
    y_true = np.concatenate(true_list, axis=0) if true_list else np.array([], dtype=np.int64)
    coords = np.concatenate(coord_list, axis=0) if coord_list else np.empty((0, 2), dtype=np.int64)

    valid_mask = y_true >= 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if y_true_valid.size == 0:
        raise ValueError("No valid labeled samples were found for evaluation.")

    confusion = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    report_text = classification_report(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    oa = accuracy_score(y_true_valid, y_pred_valid) * 100
    aa = aa * 100
    kappa = cohen_kappa_score(y_true_valid, y_pred_valid, labels=labels) * 100
    each_acc_percent = each_acc * 100
    macro_f1 = report_dict["macro avg"]["f1-score"] * 100

    metrics = {
        "dataset": dataset_name,
        "num_samples": int(y_true_valid.shape[0]),
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "per_class_accuracy": {
            class_name: float(acc)
            for class_name, acc in zip(class_names, each_acc_percent)
        },
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": confusion.astype(int),
    }

    return {
        "metrics": metrics,
        "y_true": y_true_valid,
        "y_pred": y_pred_valid,
        "coords": coords,
    }


def predict_loader(net, data_loader, device, print_freq=20):
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    start_time = time.perf_counter()
    pred_list = []
    coord_list = []
    sample_count = 0

    net.eval()
    with torch.inference_mode():
        for batch_idx, (_, x, hsi_pca, _, h, w) in enumerate(data_loader, start=1):
            hsi_pca = hsi_pca.to(device)
            x = x.to(device)
            _, outputs = net(hsi_pca.unsqueeze(1), x)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            coords_np = np.stack((h.detach().cpu().numpy(), w.detach().cpu().numpy()), axis=1)

            pred_list.append(preds)
            coord_list.append(coords_np)
            sample_count += preds.shape[0]

            if total_batches is not None and (
                batch_idx == 1 or batch_idx % print_freq == 0 or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                print(
                    f"Predict Step [{batch_idx:04d}/{total_batches:04d}] "
                    f"Samples: {sample_count} "
                    f"Elapsed: {elapsed:.2f}s"
                )

            del hsi_pca, x, outputs, preds, coords_np

    y_pred = np.concatenate(pred_list, axis=0) if pred_list else np.array([], dtype=np.int64)
    coords = np.concatenate(coord_list, axis=0) if coord_list else np.empty((0, 2), dtype=np.int64)
    return {"y_pred": y_pred, "coords": coords}


def predict_maps_loader(net, data_loader, device, image_shape, gt_map=None, print_freq=20):
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    start_time = time.perf_counter()
    pred_map = np.zeros(image_shape, dtype=np.uint8)
    visited_mask = np.zeros(image_shape, dtype=bool)
    sample_count = 0

    net.eval()
    with torch.inference_mode():
        for batch_idx, (_, x, hsi_pca, _, h, w) in enumerate(data_loader, start=1):
            hsi_pca = hsi_pca.to(device)
            x = x.to(device)
            _, outputs = net(hsi_pca.unsqueeze(1), x)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().astype(np.uint8)
            rows = h.detach().cpu().numpy().astype(np.int64)
            cols = w.detach().cpu().numpy().astype(np.int64)

            pred_map[rows, cols] = preds + 1
            visited_mask[rows, cols] = True
            sample_count += preds.shape[0]

            if total_batches is not None and (
                batch_idx == 1 or batch_idx % print_freq == 0 or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                print(
                    f"Predict Step [{batch_idx:04d}/{total_batches:04d}] "
                    f"Samples: {sample_count} "
                    f"Elapsed: {elapsed:.2f}s"
                )

            del hsi_pca, x, outputs, preds, rows, cols

    gt_masked = None
    if gt_map is not None:
        gt_masked = np.zeros_like(gt_map, dtype=np.uint8)
        gt_masked[visited_mask] = gt_map[visited_mask].astype(np.uint8)

    return {
        "pred_map": pred_map,
        "visited_mask": visited_mask,
        "gt_map": gt_masked,
    }


def save_evaluation_results(result_bundle, metrics_dir, prefix="test"):
    ensure_dir(metrics_dir)
    metrics = result_bundle["metrics"]
    confusion = metrics["confusion_matrix"]
    class_names = get_dataset_class_names(metrics["dataset"])

    json_payload = {
        "dataset": metrics["dataset"],
        "num_samples": metrics["num_samples"],
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "classification_report": metrics["classification_report_dict"],
        "confusion_matrix": confusion.tolist(),
    }
    save_json(json_payload, os.path.join(metrics_dir, f"{prefix}_metrics.json"))
    save_text(metrics["classification_report_text"], os.path.join(metrics_dir, f"{prefix}_classification_report.txt"))
    np.save(os.path.join(metrics_dir, f"{prefix}_confusion_matrix.npy"), confusion)
    np.savetxt(os.path.join(metrics_dir, f"{prefix}_confusion_matrix.csv"), confusion, fmt="%d", delimiter=",")

    rows = []
    for class_name in class_names:
        rows.append(
            {
                "class_name": class_name,
                "accuracy": metrics["per_class_accuracy"][class_name],
            }
        )
    save_csv(rows, ["class_name", "accuracy"], os.path.join(metrics_dir, f"{prefix}_per_class_accuracy.csv"))

    if prefix in ("", "test", "all"):
        save_json(json_payload, os.path.join(metrics_dir, "metrics.json"))
        save_text(metrics["classification_report_text"], os.path.join(metrics_dir, "classification_report.txt"))
        np.save(os.path.join(metrics_dir, "confusion_matrix.npy"), confusion)
        np.savetxt(os.path.join(metrics_dir, "confusion_matrix.csv"), confusion, fmt="%d", delimiter=",")
        save_csv(rows, ["class_name", "accuracy"], os.path.join(metrics_dir, "per_class_accuracy.csv"))


def summarize_metrics(metrics):
    return (
        f"OA={metrics['oa']:.4f}, "
        f"AA={metrics['aa']:.4f}, "
        f"Kappa={metrics['kappa']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )


def load_ground_truth(dataset_name, data_root="data"):
    data_info = load_dataset_info(dataset_name)
    gt_path = os.path.join(data_root, dataset_name, data_info["info"][0])
    return loadmat(gt_path)[data_info["keys"][2]]


def load_modalities(dataset_name, data_root="data"):
    data_info = load_dataset_info(dataset_name)
    dataset_dir = os.path.join(data_root, dataset_name)
    hsi = loadmat(os.path.join(dataset_dir, data_info["info"][1]))[data_info["keys"][0]]
    aux = loadmat(os.path.join(dataset_dir, data_info["info"][3]))[data_info["keys"][1]]
    return hsi, aux


def reconstruct_label_map(coords, preds, image_shape):
    label_map = np.zeros(image_shape, dtype=np.uint8)
    if coords.shape[0] == 0:
        return label_map
    rows = coords[:, 0].astype(np.int64)
    cols = coords[:, 1].astype(np.int64)
    label_map[rows, cols] = preds.astype(np.uint8) + 1
    return label_map


def mask_ground_truth(gt_map, coords):
    masked = np.zeros_like(gt_map, dtype=np.uint8)
    if coords.shape[0] == 0:
        return masked
    rows = coords[:, 0].astype(np.int64)
    cols = coords[:, 1].astype(np.int64)
    masked[rows, cols] = gt_map[rows, cols].astype(np.uint8)
    return masked


def colorize_label_map(label_map, dataset_name):
    palette = np.asarray(get_dataset_palette(dataset_name), dtype=np.uint8)
    rgb = palette[label_map]
    return rgb.astype(np.uint8)


def save_label_map_image(label_map, dataset_name, save_path):
    rgb = colorize_label_map(label_map, dataset_name)
    Image.fromarray(rgb).save(save_path)


def _percentile_normalize(array, low=2, high=98):
    array = array.astype(np.float32)
    lo = np.percentile(array, low)
    hi = np.percentile(array, high)
    if hi <= lo:
        hi = array.max()
        lo = array.min()
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    array = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def _default_rgb_bands(num_bands):
    if num_bands < 3:
        raise ValueError("HSI must have at least 3 bands to create a pseudo-RGB view.")
    return [
        min(num_bands - 1, max(0, int(num_bands * 0.75))),
        min(num_bands - 1, max(0, int(num_bands * 0.5))),
        min(num_bands - 1, max(0, int(num_bands * 0.25))),
    ]


def hsi_to_rgb_image(hsi):
    band_ids = _default_rgb_bands(hsi.shape[2])
    rgb = np.stack([hsi[:, :, idx] for idx in band_ids], axis=-1)
    return _percentile_normalize(rgb)


def lidar_to_grayscale_image(lidar):
    if lidar.ndim == 3:
        lidar = lidar[:, :, 0]
    gray = _percentile_normalize(lidar)
    return np.stack([gray, gray, gray], axis=-1)


def save_input_view_artifacts(dataset_name, hsi, aux, figures_dir, prefix=""):
    ensure_dir(figures_dir)
    name_prefix = f"{prefix}_" if prefix else ""

    hsi_rgb = hsi_to_rgb_image(hsi)
    aux_rgb = lidar_to_grayscale_image(aux)
    pred_img = Image.fromarray(hsi_rgb)
    aux_img = Image.fromarray(aux_rgb)

    hsi_path = os.path.join(figures_dir, f"{name_prefix}hsi_rgb.png")
    aux_path = os.path.join(figures_dir, f"{name_prefix}aux_view.png")
    pred_img.save(hsi_path)
    aux_img.save(aux_path)

    panel = Image.new("RGB", (pred_img.width + aux_img.width, max(pred_img.height, aux_img.height)))
    panel.paste(pred_img, (0, 0))
    panel.paste(aux_img, (pred_img.width, 0))
    panel.save(os.path.join(figures_dir, f"{name_prefix}input_comparison.png"))
    return hsi_rgb, aux_rgb


def save_prediction_artifacts(dataset_name, pred_map, gt_map, predictions_dir, figures_dir, prefix=""):
    ensure_dir(predictions_dir)
    ensure_dir(figures_dir)
    name_prefix = f"{prefix}_" if prefix else ""

    np.save(os.path.join(predictions_dir, f"{name_prefix}prediction_map.npy"), pred_map)
    savemat(
        os.path.join(predictions_dir, f"{name_prefix}prediction_map.mat"),
        {"prediction_map": pred_map},
    )

    save_label_map_image(pred_map, dataset_name, os.path.join(figures_dir, f"{name_prefix}prediction_map.png"))
    save_label_map_image(gt_map, dataset_name, os.path.join(figures_dir, f"{name_prefix}gt_map.png"))


def save_four_panel_figure(dataset_name, pred_map, gt_map, hsi_rgb, aux_rgb, figures_dir, prefix=""):
    ensure_dir(figures_dir)
    name_prefix = f"{prefix}_" if prefix else ""

    pred_rgb = colorize_label_map(pred_map, dataset_name)
    gt_rgb = colorize_label_map(gt_map, dataset_name)

    pred_img = Image.fromarray(pred_rgb)
    gt_img = Image.fromarray(gt_rgb)
    hsi_img = Image.fromarray(hsi_rgb)
    aux_img = Image.fromarray(aux_rgb)

    panel_width = pred_img.width
    panel_height = pred_img.height
    title_height = 28
    canvas = Image.new("RGB", (panel_width * 4, panel_height + title_height), color=(255, 255, 255))

    title_bar = Image.new("RGB", (panel_width, title_height), color=(245, 245, 245))
    labels = [
        ("Prediction", pred_img),
        ("Ground Truth", gt_img),
        ("HSI RGB", hsi_img),
        ("LiDAR", aux_img),
    ]
    for idx, (_, img) in enumerate(labels):
        canvas.paste(title_bar, (idx * panel_width, 0))
        canvas.paste(img, (idx * panel_width, title_height))

    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        for idx, (title, _) in enumerate(labels):
            draw.text((idx * panel_width + 8, 6), title, fill=(0, 0, 0))
    except Exception:
        pass

    canvas.save(os.path.join(figures_dir, f"{name_prefix}four_panel.png"))


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def createDatasetReport(net, data, dataset_name, device):
    result_bundle = evaluate_model(net=net, data_loader=data, dataset_name=dataset_name, device=device)
    metrics = result_bundle["metrics"]
    return metrics["oa"], metrics["aa"], metrics["kappa"], np.asarray(list(metrics["per_class_accuracy"].values()))

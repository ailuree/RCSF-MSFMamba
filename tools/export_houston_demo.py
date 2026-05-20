import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn import preprocessing

import utility
from model.MSFMamba import Net
from setting.dataLoader import HXDataset, applyPCA


def parse_args():
    parser = argparse.ArgumentParser("Export dense ROI demo assets for Houston datasets")
    parser.add_argument("--dataset", type=str, default="Houston2013")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="./checkpoints/")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_work", type=int, default=0)
    return parser.parse_args()


def build_dense_coords(image_h, image_w, x, y, width, height, pad):
    x0 = max(pad, x)
    y0 = max(pad, y)
    x1 = min(image_w - pad, x + width)
    y1 = min(image_h - pad, y + height)
    coords = [[row, col] for row in range(y0, y1) for col in range(x0, x1)]
    return np.asarray(coords, dtype=np.int64), (x0, y0, x1 - x0, y1 - y0)


def hex_color(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def save_rgb_image(array, path):
    Image.fromarray(array.astype(np.uint8)).save(path)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_info = utility.load_dataset_info(args.dataset)
    pad = data_info["window_size"] // 2

    hsi, aux = utility.load_modalities(args.dataset)
    gt = utility.load_ground_truth(args.dataset)

    coords, roi = build_dense_coords(
        image_h=hsi.shape[0],
        image_w=hsi.shape[1],
        x=args.x,
        y=args.y,
        width=args.width,
        height=args.height,
        pad=pad,
    )
    if coords.size == 0:
        raise ValueError("ROI is empty after border-safe adjustment.")

    hsi_pca = applyPCA(hsi, data_info["pca_num"])
    hsi_pca_scaled = preprocessing.scale(hsi_pca.reshape(np.prod(hsi_pca.shape[:2]), hsi_pca.shape[2]))
    hsi_pca = hsi_pca_scaled.reshape(hsi_pca.shape[0], hsi_pca.shape[1], hsi_pca.shape[2])

    if aux.ndim == 2:
        aux_scaled = preprocessing.scale(aux.reshape(np.prod(aux.shape[:2])))
        aux_scaled = aux_scaled.reshape(aux.shape[0], aux.shape[1])
    else:
        aux_scaled = preprocessing.scale(aux.reshape(np.prod(aux.shape[:2]), aux.shape[2]))
        aux_scaled = aux_scaled.reshape(aux.shape[0], aux.shape[1], aux.shape[2])

    dense_set = HXDataset(hsi=hsi, Xdata=aux_scaled, hsi_pca=hsi_pca, index=coords, gt=gt, windowSize=data_info["window_size"])
    dense_loader = torch.utils.data.DataLoader(
        dense_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_work,
        pin_memory=True,
        drop_last=False,
    )

    _, checkpoint_path = utility.resolve_checkpoint(
        save_root=args.save_root,
        dataset=args.dataset,
        run_name=args.run_name,
        prefer="best",
    )
    model = Net(args.dataset).to(device)
    utility.load_checkpoint(model, checkpoint_path, device)
    model.eval()

    roi_x, roi_y, roi_w, roi_h = roi
    pred_map = np.zeros((roi_h, roi_w), dtype=np.uint8)
    conf_map = np.zeros((roi_h, roi_w), dtype=np.float32)

    with torch.inference_mode():
        for _, aux_batch, hsi_pca_batch, _, rows, cols in dense_loader:
            aux_batch = aux_batch.to(device, non_blocking=True)
            hsi_pca_batch = hsi_pca_batch.to(device, non_blocking=True)
            _, logits = model(hsi_pca_batch.unsqueeze(1), aux_batch)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            rows = rows.numpy()
            cols = cols.numpy()
            preds = preds.cpu().numpy()
            confs = confs.cpu().numpy()
            pred_map[rows - roi_y, cols - roi_x] = preds.astype(np.uint8) + 1
            conf_map[rows - roi_y, cols - roi_x] = confs.astype(np.float32)

    roi_hsi = utility.hsi_to_rgb_image(hsi)[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_aux = utility.lidar_to_grayscale_image(aux)[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_gt = gt[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].astype(np.uint8)
    roi_pred_rgb = utility.colorize_label_map(pred_map, args.dataset)
    roi_gt_rgb = utility.colorize_label_map(roi_gt, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    save_rgb_image(roi_hsi, os.path.join(args.output_dir, "roi_hsi.png"))
    save_rgb_image(roi_aux, os.path.join(args.output_dir, "roi_aux.png"))
    save_rgb_image(roi_pred_rgb, os.path.join(args.output_dir, "roi_prediction.png"))
    save_rgb_image(roi_gt_rgb, os.path.join(args.output_dir, "roi_gt.png"))

    classes = utility.get_dataset_class_names(args.dataset)
    palette = utility.get_dataset_palette(args.dataset)
    class_items = [
        {"id": idx, "name": classes[idx - 1], "color": hex_color(palette[idx])}
        for idx in range(1, len(classes) + 1)
    ]

    unique, counts = np.unique(pred_map[pred_map > 0], return_counts=True)
    ranked = sorted(
        [
            {
                "id": int(label),
                "name": classes[label - 1],
                "color": hex_color(palette[label]),
                "count": int(count),
                "ratio": round(float(count / pred_map.size), 4),
            }
            for label, count in zip(unique.tolist(), counts.tolist())
        ],
        key=lambda item: item["count"],
        reverse=True,
    )

    metadata = {
        "dataset": args.dataset,
        "run_name": args.run_name,
        "checkpoint": checkpoint_path,
        "roi": {"x": roi_x, "y": roi_y, "width": roi_w, "height": roi_h},
        "classes": class_items,
        "top_classes": ranked[:8],
        "prediction_labels": pred_map.tolist(),
        "confidence_map": np.round(conf_map, 4).tolist(),
        "ground_truth_labels": roi_gt.tolist(),
    }
    with open(os.path.join(args.output_dir, "roi_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved dense ROI demo assets to {args.output_dir}")
    print(f"ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Top classes: {[item['name'] for item in ranked[:5]]}")


if __name__ == "__main__":
    main()

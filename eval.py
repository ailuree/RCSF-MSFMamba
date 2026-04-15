import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix

import utility
from model.MSFMamba import Net
from setting.dataLoader import get_loader
from setting.robustness import (
    build_degraded_batch,
    degrade_aux_batch,
    degrade_hsi_batch,
    summarize_degradation_stats,
)


def build_parser():
    parser = argparse.ArgumentParser("Standalone evaluation for MSFMamba baselines")
    parser.add_argument("--dataset", type=str, default="Houston2013")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint path, default uses latest best.pth")
    parser.add_argument("--run_name", type=str, default="", help="optional run directory name under checkpoints/<dataset>")
    parser.add_argument("--save_path", type=str, default="./checkpoints/", help="root checkpoint directory")
    parser.add_argument("--save_dir", type=str, default="", help="optional output directory for metrics")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_work", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "trntst", "all"])
    parser.add_argument("--useval", type=int, default=0)
    parser.add_argument("--eval_split", type=str, default="small", choices=["small", "full"])
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument(
        "--robust_eval_mode",
        type=str,
        default="none",
        choices=["none", "drop_hsi", "drop_aux", "noise", "combined"],
    )
    parser.add_argument("--modality_dropout_prob", type=float, default=0.2)
    parser.add_argument("--hsi_dropout_prob", type=float, default=0.15)
    parser.add_argument("--hsi_noise_std", type=float, default=0.05)
    parser.add_argument("--aux_noise_std", type=float, default=0.03)
    return parser


def select_loader(split, train_loader, test_loader, trntst_loader, all_loader):
    mapping = {
        "train": train_loader,
        "test": test_loader,
        "trntst": trntst_loader,
        "all": all_loader,
    }
    return mapping[split]


def to_serializable_metrics(metrics):
    return {
        "dataset": metrics["dataset"],
        "num_samples": metrics["num_samples"],
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": metrics["per_class_accuracy"],
    }


def apply_eval_degradation(hsi, hsi_pca, aux, args):
    mode = args.robust_eval_mode
    if mode == "none":
        return hsi, hsi_pca, aux, None
    if mode == "drop_hsi":
        return torch.zeros_like(hsi), torch.zeros_like(hsi_pca), aux.clone(), {
            "drop_hsi_ratio": 1.0,
            "drop_aux_ratio": 0.0,
            "hsi_noise_applied_ratio": 0.0,
            "aux_noise_applied_ratio": 0.0,
            "hsi_band_dropout_ratio": 0.0,
        }
    if mode == "drop_aux":
        return hsi.clone(), hsi_pca.clone(), torch.zeros_like(aux), {
            "drop_hsi_ratio": 0.0,
            "drop_aux_ratio": 1.0,
            "hsi_noise_applied_ratio": 0.0,
            "aux_noise_applied_ratio": 0.0,
            "hsi_band_dropout_ratio": 0.0,
        }
    if mode == "noise":
        hsi_deg, hsi_pca_deg, hsi_stats = degrade_hsi_batch(
            hsi,
            hsi_pca,
            hsi_dropout_prob=0.0,
            hsi_noise_std=args.hsi_noise_std,
        )
        aux_deg, aux_stats = degrade_aux_batch(aux, aux_noise_std=args.aux_noise_std)
        stats = {
            "drop_hsi_ratio": 0.0,
            "drop_aux_ratio": 0.0,
            "hsi_noise_applied_ratio": hsi_stats["hsi_noise_applied_ratio"],
            "aux_noise_applied_ratio": aux_stats["aux_noise_applied_ratio"],
            "hsi_band_dropout_ratio": 0.0,
        }
        return hsi_deg, hsi_pca_deg, aux_deg, stats

    degraded_batch = build_degraded_batch(
        hsi=hsi,
        hsi_pca=hsi_pca,
        aux=aux,
        modality_dropout_prob=args.modality_dropout_prob,
        hsi_dropout_prob=args.hsi_dropout_prob,
        hsi_noise_std=args.hsi_noise_std,
        aux_noise_std=args.aux_noise_std,
    )
    return (
        degraded_batch["hsi_deg"],
        degraded_batch["hsi_pca_deg"],
        degraded_batch["aux_deg"],
        degraded_batch["stats"],
    )


def evaluate_with_mode(net, data_loader, dataset_name, device, print_freq, args):
    class_names = utility.get_dataset_class_names(dataset_name)
    labels = list(range(len(class_names)))
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None

    net.eval()
    pred_list = []
    true_list = []
    coord_list = []
    stat_accumulator = {
        "drop_hsi_ratio": 0.0,
        "drop_aux_ratio": 0.0,
        "hsi_noise_applied_ratio": 0.0,
        "aux_noise_applied_ratio": 0.0,
        "hsi_band_dropout_ratio": 0.0,
    }
    stat_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (hsi, x, hsi_pca, test_labels, h, w) in enumerate(data_loader, start=1):
            hsi = hsi.to(device)
            hsi_pca = hsi_pca.to(device)
            x = x.to(device)
            hsi_deg, hsi_pca_deg, x_deg, stats = apply_eval_degradation(hsi, hsi_pca, x, args)

            _, outputs = net(hsi_pca_deg.unsqueeze(1), x_deg)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels_np = test_labels.detach().cpu().numpy()
            coords_np = np.stack((h.detach().cpu().numpy(), w.detach().cpu().numpy()), axis=1)

            pred_list.append(preds)
            true_list.append(labels_np)
            coord_list.append(coords_np)
            if stats is not None:
                for key in stat_accumulator:
                    stat_accumulator[key] += float(stats.get(key, 0.0))
                stat_count += 1

            if total_batches is not None and (
                batch_idx == 1 or batch_idx % print_freq == 0 or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - start_time
                message = (
                    f"Eval Step [{batch_idx:04d}/{total_batches:04d}] "
                    f"Samples: {sum(arr.shape[0] for arr in pred_list)} "
                    f"Elapsed: {elapsed:.2f}s"
                )
                if stats is not None:
                    message += " " + summarize_degradation_stats(stats)
                print(message)

    y_pred = np.concatenate(pred_list, axis=0) if pred_list else np.array([], dtype=np.int64)
    y_true = np.concatenate(true_list, axis=0) if true_list else np.array([], dtype=np.int64)
    coords = np.concatenate(coord_list, axis=0) if coord_list else np.empty((0, 2), dtype=np.int64)
    valid_mask = y_true >= 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if y_true_valid.size == 0:
        raise ValueError("No valid labeled samples were found for evaluation.")

    confusion = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    each_acc, aa = utility.AA_andEachClassAccuracy(confusion)
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
    metrics = {
        "dataset": dataset_name,
        "num_samples": int(y_true_valid.shape[0]),
        "oa": float(accuracy_score(y_true_valid, y_pred_valid) * 100),
        "aa": float(aa * 100),
        "kappa": float(cohen_kappa_score(y_true_valid, y_pred_valid, labels=labels) * 100),
        "macro_f1": float(report_dict["macro avg"]["f1-score"] * 100),
        "per_class_accuracy": {
            class_name: float(acc)
            for class_name, acc in zip(class_names, each_acc * 100)
        },
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": confusion.astype(int),
    }
    if stat_count > 0:
        metrics["degradation_stats"] = {
            key: stat_accumulator[key] / stat_count
            for key in stat_accumulator
        }

    return {
        "metrics": metrics,
        "y_true": y_true_valid,
        "y_pred": y_pred_valid,
        "coords": coords,
    }


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_dir, checkpoint_path = utility.resolve_checkpoint(
        save_root=args.save_path,
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
        prefer="best",
    )

    train_loader, test_loader, trntst_loader, all_loader, _, _, _ = get_loader(
        dataset=args.dataset,
        batchsize=args.batchsize,
        num_workers=args.num_work,
        useval=args.useval,
        pin_memory=True,
        eval_split=args.eval_split,
    )
    data_loader = select_loader(args.split, train_loader, test_loader, trntst_loader, all_loader)

    model = Net(args.dataset).to(device)
    utility.load_checkpoint(model, checkpoint_path, device)

    result_bundle = evaluate_with_mode(
        net=model,
        data_loader=data_loader,
        dataset_name=args.dataset,
        device=device,
        print_freq=max(args.print_freq, 1),
        args=args,
    )
    metrics_dir = args.save_dir or os.path.join(run_dir, "metrics")
    result_prefix = args.split if args.robust_eval_mode == "none" else f"{args.split}_{args.robust_eval_mode}"
    utility.save_evaluation_results(result_bundle, metrics_dir, prefix=result_prefix)

    serializable_metrics = to_serializable_metrics(result_bundle["metrics"])
    if "degradation_stats" in result_bundle["metrics"]:
        serializable_metrics["degradation_stats"] = result_bundle["metrics"]["degradation_stats"]
    utility.save_json(serializable_metrics, os.path.join(metrics_dir, f"{result_prefix}_metrics_summary.json"))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {metrics_dir}")
    print(f"Robust eval mode: {args.robust_eval_mode}")
    print(utility.summarize_metrics(result_bundle["metrics"]))

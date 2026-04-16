import argparse
import csv
import json
import os
from statistics import mean, pstdev


METHOD_ORDER = ["baseline", "gate", "cross", "robust", "full_model"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "gate": "Reliability Gate",
    "cross": "Cross-State",
    "robust": "Robust Training",
    "full_model": "Full Model v1",
}
ROBUST_METHODS = {"robust", "full_model"}
ROBUSTNESS_MODES = {
    "normal": "test_metrics_summary.json",
    "drop_hsi": "test_drop_hsi_metrics_summary.json",
    "drop_aux": "test_drop_aux_metrics_summary.json",
    "noise": "test_noise_metrics_summary.json",
    "combined": "test_combined_metrics_summary.json",
}
MAIN_METRIC_FIELDS = ["oa", "aa", "kappa", "macro_f1"]


def parse_args():
    parser = argparse.ArgumentParser("Summarize multi-seed experiment results.")
    parser.add_argument("--dataset", required=True, choices=["Houston2013", "Houston2018"])
    parser.add_argument("--variant", default="", choices=["", "small", "full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--checkpoints_root", default="./checkpoints")
    return parser.parse_args()


def require_variant(dataset, variant):
    if dataset == "Houston2018" and not variant:
        raise ValueError("Houston2018 requires --variant small or --variant full.")
    if dataset == "Houston2013" and variant:
        raise ValueError("Houston2013 does not use --variant.")


def run_prefixes(dataset, variant):
    if dataset == "Houston2013":
        return {
            "baseline": "baseline_h2013",
            "gate": "gate_h2013",
            "cross": "cross_h2013",
            "robust": "robust_h2013",
            "full_model": "full_model_h2013",
        }
    suffix = f"h2018_{variant}"
    return {
        "baseline": f"baseline_{suffix}",
        "gate": f"gate_{suffix}",
        "cross": f"cross_{suffix}",
        "robust": f"robust_{suffix}",
        "full_model": f"full_model_{suffix}",
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_csv(rows, headers, path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def metric_path(run_dir, filename):
    return os.path.join(run_dir, "metrics", filename)


def read_main_metrics(run_dir):
    path = metric_path(run_dir, "test_metrics_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing main metrics file: {path}")
    return load_json(path)


def read_robustness_metrics(run_dir):
    results = {}
    for mode, filename in ROBUSTNESS_MODES.items():
        path = metric_path(run_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing robustness metrics file: {path}")
        results[mode] = load_json(path)
    return results


def summarize_metric_values(values):
    return {
        "mean": mean(values),
        "std": pstdev(values),
    }


def main():
    args = parse_args()
    require_variant(args.dataset, args.variant)

    dataset_root = os.path.join(args.checkpoints_root, args.dataset)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset checkpoints directory not found: {dataset_root}")

    output_dir = os.path.join(dataset_root, "完整实验结果汇总", "seed_summary")
    os.makedirs(output_dir, exist_ok=True)

    prefixes = run_prefixes(args.dataset, args.variant)
    variant_name = args.variant or ""

    main_rows = []
    robustness_rows = []
    summary = {
        "dataset": args.dataset,
        "variant": variant_name,
        "seeds": args.seeds,
        "methods": {},
    }

    for method in METHOD_ORDER:
        prefix = prefixes[method]
        method_summary = {
            "label": METHOD_LABELS[method],
            "prefix": prefix,
            "runs": [],
        }
        metric_values = {field: [] for field in MAIN_METRIC_FIELDS}

        for seed in args.seeds:
            run_name = f"{prefix}_s{seed}"
            run_dir = os.path.join(dataset_root, run_name)
            if not os.path.isdir(run_dir):
                raise FileNotFoundError(f"Missing run directory: {run_dir}")

            main_metrics = read_main_metrics(run_dir)
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "run_name": run_name,
                "oa": float(main_metrics["oa"]),
                "aa": float(main_metrics["aa"]),
                "kappa": float(main_metrics["kappa"]),
                "macro_f1": float(main_metrics["macro_f1"]),
            }
            main_rows.append(row)
            method_summary["runs"].append(row)
            for field in MAIN_METRIC_FIELDS:
                metric_values[field].append(row[field])

            if method in ROBUST_METHODS:
                robustness_metrics = read_robustness_metrics(run_dir)
                for mode in ["normal", "drop_hsi", "drop_aux", "noise", "combined"]:
                    robustness_rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "seed": seed,
                            "run_name": run_name,
                            "mode": mode,
                            "oa": float(robustness_metrics[mode]["oa"]),
                        }
                    )

        method_summary["main_mean_std"] = {}
        for field in MAIN_METRIC_FIELDS:
            stats = summarize_metric_values(metric_values[field])
            method_summary["main_mean_std"][field] = {
                "mean": stats["mean"],
                "std": stats["std"],
            }

        if method in ROBUST_METHODS:
            method_summary["robustness_mean_std"] = {}
            for mode in ["normal", "drop_hsi", "drop_aux", "noise", "combined"]:
                values = [
                    row["oa"]
                    for row in robustness_rows
                    if row["method"] == method and row["mode"] == mode
                ]
                stats = summarize_metric_values(values)
                method_summary["robustness_mean_std"][mode] = {
                    "oa_mean": stats["mean"],
                    "oa_std": stats["std"],
                }

        summary["methods"][method] = method_summary

    main_mean_std_rows = []
    for method in METHOD_ORDER:
        stats = summary["methods"][method]["main_mean_std"]
        main_mean_std_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "oa_mean": stats["oa"]["mean"],
                "oa_std": stats["oa"]["std"],
                "aa_mean": stats["aa"]["mean"],
                "aa_std": stats["aa"]["std"],
                "kappa_mean": stats["kappa"]["mean"],
                "kappa_std": stats["kappa"]["std"],
                "macro_f1_mean": stats["macro_f1"]["mean"],
                "macro_f1_std": stats["macro_f1"]["std"],
            }
        )

    robustness_mean_std_rows = []
    for method in METHOD_ORDER:
        if method not in ROBUST_METHODS:
            continue
        stats = summary["methods"][method]["robustness_mean_std"]
        for mode in ["normal", "drop_hsi", "drop_aux", "noise", "combined"]:
            robustness_mean_std_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mode": mode,
                    "oa_mean": stats[mode]["oa_mean"],
                    "oa_std": stats[mode]["oa_std"],
                }
            )

    save_json(summary, os.path.join(output_dir, "summary.json"))
    save_csv(
        main_rows,
        ["method", "method_label", "seed", "run_name", "oa", "aa", "kappa", "macro_f1"],
        os.path.join(output_dir, "main_results_by_seed.csv"),
    )
    save_csv(
        main_mean_std_rows,
        [
            "method",
            "method_label",
            "oa_mean",
            "oa_std",
            "aa_mean",
            "aa_std",
            "kappa_mean",
            "kappa_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
        os.path.join(output_dir, "main_results_mean_std.csv"),
    )
    save_csv(
        robustness_rows,
        ["method", "method_label", "seed", "run_name", "mode", "oa"],
        os.path.join(output_dir, "robustness_by_seed.csv"),
    )
    save_csv(
        robustness_mean_std_rows,
        ["method", "method_label", "mode", "oa_mean", "oa_std"],
        os.path.join(output_dir, "robustness_mean_std.csv"),
    )

    print(f"Dataset: {args.dataset}")
    if args.variant:
        print(f"Variant: {args.variant}")
    print(f"Seeds: {args.seeds}")
    print(f"Output dir: {output_dir}")
    print("Saved:")
    print("- summary.json")
    print("- main_results_by_seed.csv")
    print("- main_results_mean_std.csv")
    print("- robustness_by_seed.csv")
    print("- robustness_mean_std.csv")


if __name__ == "__main__":
    main()

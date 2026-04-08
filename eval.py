import argparse
import os

import torch

import utility
from model.MSFMamba import Net
from setting.dataLoader import get_loader


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

    result_bundle = utility.evaluate_model(
        net=model,
        data_loader=data_loader,
        dataset_name=args.dataset,
        device=device,
        print_freq=max(args.print_freq, 1),
    )
    metrics_dir = args.save_dir or os.path.join(run_dir, "metrics")
    utility.save_evaluation_results(result_bundle, metrics_dir, prefix=args.split)

    serializable_metrics = to_serializable_metrics(result_bundle["metrics"])
    utility.save_json(serializable_metrics, os.path.join(metrics_dir, f"{args.split}_metrics_summary.json"))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {metrics_dir}")
    print(utility.summarize_metrics(result_bundle["metrics"]))

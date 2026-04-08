import argparse
import os

import torch

import utility
from model.MSFMamba import Net
from setting.dataLoader import get_loader


def build_parser():
    parser = argparse.ArgumentParser("Full-scene prediction and visualization for MSFMamba baselines")
    parser.add_argument("--dataset", type=str, default="Houston2013")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint path, default uses latest best.pth")
    parser.add_argument("--run_name", type=str, default="", help="optional run directory name under checkpoints/<dataset>")
    parser.add_argument("--save_path", type=str, default="./checkpoints/", help="root checkpoint directory")
    parser.add_argument("--save_dir", type=str, default="", help="optional base output directory")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_work", type=int, default=0)
    parser.add_argument("--split", type=str, default="all", choices=["test", "train", "trntst", "all"])
    parser.add_argument("--useval", type=int, default=0)
    parser.add_argument("--eval_split", type=str, default="small", choices=["small", "full"])
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument(
        "--save_input_views",
        type=int,
        default=0,
        help="set to 1 to additionally save HSI pseudo-RGB, auxiliary modality view, and a comparison panel",
    )
    return parser


def select_loader(split, train_loader, test_loader, trntst_loader, all_loader):
    mapping = {
        "train": train_loader,
        "test": test_loader,
        "trntst": trntst_loader,
        "all": all_loader,
    }
    return mapping[split]


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
    base_output_dir = args.save_dir or run_dir
    figures_dir = os.path.join(base_output_dir, "figures")
    predictions_dir = os.path.join(base_output_dir, "predictions")

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

    prediction_bundle = utility.predict_loader(
        net=model,
        data_loader=data_loader,
        device=device,
        print_freq=max(args.print_freq, 1),
    )

    gt_full = utility.load_ground_truth(args.dataset)
    pred_map = utility.reconstruct_label_map(prediction_bundle["coords"], prediction_bundle["y_pred"], gt_full.shape)
    gt_map = utility.mask_ground_truth(gt_full, prediction_bundle["coords"])
    prefix = "" if args.split == "all" else args.split
    utility.save_prediction_artifacts(
        dataset_name=args.dataset,
        pred_map=pred_map,
        gt_map=gt_map,
        predictions_dir=predictions_dir,
        figures_dir=figures_dir,
        prefix=prefix,
    )

    if args.save_input_views:
        hsi, aux = utility.load_modalities(args.dataset)
        hsi_rgb, aux_rgb = utility.save_input_view_artifacts(
            dataset_name=args.dataset,
            hsi=hsi,
            aux=aux,
            figures_dir=figures_dir,
            prefix=prefix,
        )
        utility.save_four_panel_figure(
            dataset_name=args.dataset,
            pred_map=pred_map,
            gt_map=gt_map,
            hsi_rgb=hsi_rgb,
            aux_rgb=aux_rgb,
            figures_dir=figures_dir,
            prefix=prefix,
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Figures dir: {figures_dir}")
    print(f"Predictions dir: {predictions_dir}")
    print(f"Saved {args.split} prediction map for {args.dataset}")

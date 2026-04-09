import logging
import os
import time

import torch
import torch.nn.functional as F

import utility
from model.MSFMamba import Net
from setting.dataLoader import get_loader
from setting.options import opt
from setting.robustness import build_degraded_batch, summarize_degradation_stats
from setting.utils import compute_accuracy, random_seed_setting

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

random_seed_setting(6)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
run_dirs = utility.create_run_structure(opt.save_path, opt.dataset, opt.run_name)
history_path = os.path.join(run_dirs["metrics_dir"], "epoch_history.csv")
best_metrics_path = os.path.join(run_dirs["metrics_dir"], "best_metrics.json")
last_metrics_path = os.path.join(run_dirs["metrics_dir"], "last_metrics.json")

logging.basicConfig(
    filename=os.path.join(run_dirs["logs_dir"], "train.log"),
    format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
    level=logging.INFO,
    filemode="a",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)

logging.info("********************start train!********************")
logging.info("Run directory: %s", run_dirs["run_dir"])
logging.info("Config--epoch:%s; lr:%s; batch_size:%s;", opt.epoch, opt.lr, opt.batchsize)
utility.save_json(vars(opt), os.path.join(run_dirs["metrics_dir"], "train_config.json"))

train_loader, test_loader, trntst_loader, all_loader, train_num, val_num, trntst_num = get_loader(
    dataset=opt.dataset,
    batchsize=opt.batchsize,
    num_workers=opt.num_work,
    useval=opt.useval,
    pin_memory=True,
    eval_split=opt.eval_split,
)

logging.info(
    "Loading data, including %s training images and %s validation images and %s train_test images",
    train_num,
    val_num,
    trntst_num,
)

model = Net(opt.dataset).to(device)
optimizer = torch.optim.Adam(model.parameters(), opt.lr)
criterion = torch.nn.CrossEntropyLoss().to(device)

best_acc = opt.best_acc
best_epoch = opt.best_epoch


def metric_summary_dict(metrics):
    return {
        "dataset": metrics["dataset"],
        "num_samples": metrics["num_samples"],
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": metrics["per_class_accuracy"],
    }


def save_checkpoint(model_obj, optimizer_obj, checkpoint_path, epoch, metrics=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model_obj.state_dict(),
        "optimizer_state_dict": optimizer_obj.state_dict(),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, checkpoint_path)


def consistency_kl_loss(clean_logits, degraded_logits):
    clean_prob = F.softmax(clean_logits.detach(), dim=1)
    degraded_log_prob = F.log_softmax(degraded_logits, dim=1)
    return F.kl_div(degraded_log_prob, clean_prob, reduction="batchmean")


def train_one_epoch(data_loader, model_obj, optimizer_obj, epoch):
    model_obj.train()
    loss_all = 0.0
    iteration = len(data_loader)
    acc = 0.0
    num = 0
    clean_loss_all = 0.0
    deg_loss_all = 0.0
    cons_loss_all = 0.0
    drop_hsi_sum = 0.0
    drop_aux_sum = 0.0
    hsi_noise_sum = 0.0
    aux_noise_sum = 0.0
    band_drop_sum = 0.0

    for i, (hsi, xdata, hsi_pca, gt, _, _) in enumerate(data_loader, start=1):
        optimizer_obj.zero_grad()
        hsi = hsi.to(device)
        xdata = xdata.to(device)
        hsi_pca = hsi_pca.to(device)
        gt = gt.to(device)

        _, clean_outputs = model_obj(hsi_pca.unsqueeze(1), xdata)
        clean_loss = criterion(clean_outputs, gt)
        degraded_loss = torch.zeros((), device=device)
        consistency_loss = torch.zeros((), device=device)
        degradation_stats = None

        if opt.robust_train:
            degraded_batch = build_degraded_batch(
                hsi=hsi,
                hsi_pca=hsi_pca,
                aux=xdata,
                modality_dropout_prob=opt.modality_dropout_prob,
                hsi_dropout_prob=opt.hsi_dropout_prob,
                hsi_noise_std=opt.hsi_noise_std,
                aux_noise_std=opt.aux_noise_std,
            )
            _, degraded_outputs = model_obj(
                degraded_batch["hsi_pca_deg"].unsqueeze(1),
                degraded_batch["aux_deg"],
            )
            degraded_loss = criterion(degraded_outputs, gt)
            if opt.lambda_cons > 0:
                consistency_loss = consistency_kl_loss(clean_outputs, degraded_outputs)
            degradation_stats = degraded_batch["stats"]

        loss = clean_loss + opt.lambda_deg * degraded_loss + opt.lambda_cons * consistency_loss
        loss.backward()
        optimizer_obj.step()

        loss_all += loss.item()
        clean_loss_all += clean_loss.item()
        deg_loss_all += degraded_loss.item()
        cons_loss_all += consistency_loss.item()
        acc += compute_accuracy(clean_outputs, gt) * len(gt)
        num += len(gt)
        if degradation_stats is not None:
            drop_hsi_sum += degradation_stats["drop_hsi_ratio"]
            drop_aux_sum += degradation_stats["drop_aux_ratio"]
            hsi_noise_sum += degradation_stats["hsi_noise_applied_ratio"]
            aux_noise_sum += degradation_stats["aux_noise_applied_ratio"]
            band_drop_sum += degradation_stats.get("hsi_band_dropout_ratio", 0.0)

        if opt.print_freq > 0 and (i == 1 or i % opt.print_freq == 0 or i == iteration):
            message = (
                f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] "
                f"Step [{i:04d}/{iteration:04d}] "
                f"Loss: {loss.item():.4f}"
            )
            if opt.robust_train:
                message += (
                    f" CleanCE: {clean_loss.item():.4f}"
                    f" DegCE: {degraded_loss.item():.4f}"
                    f" Cons: {consistency_loss.item():.4f}"
                )
                if degradation_stats is not None:
                    message += " " + summarize_degradation_stats(degradation_stats)
            print(message)

        if opt.max_train_batches > 0 and i >= opt.max_train_batches:
            print(f"Stop early after {i} train batches for smoke test.")
            break

    loss_avg = loss_all / i
    acc_avg = acc / max(num, 1)
    epoch_summary = {
        "train_loss": loss_avg,
        "train_acc": acc_avg * 100,
        "clean_loss": clean_loss_all / i,
        "deg_loss": deg_loss_all / i,
        "cons_loss": cons_loss_all / i,
    }
    if opt.robust_train:
        epoch_summary.update(
            {
                "drop_hsi_ratio": drop_hsi_sum / i,
                "drop_aux_ratio": drop_aux_sum / i,
                "hsi_noise_applied_ratio": hsi_noise_sum / i,
                "aux_noise_applied_ratio": aux_noise_sum / i,
                "hsi_band_dropout_ratio": band_drop_sum / i,
            }
        )
    logging.info(
        "Epoch [%03d/%03d], Loss_train_avg: %.4f, acc_avg: %.4f, clean_loss: %.4f, deg_loss: %.4f, cons_loss: %.4f, drop_hsi: %.4f, drop_aux: %.4f",
        epoch,
        opt.epoch,
        loss_avg,
        acc_avg,
        epoch_summary["clean_loss"],
        epoch_summary["deg_loss"],
        epoch_summary["cons_loss"],
        epoch_summary.get("drop_hsi_ratio", 0.0),
        epoch_summary.get("drop_aux_ratio", 0.0),
    )
    summary_message = (
        f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] Done, "
        f"Loss_avg: {loss_avg:.4f}, Acc_avg: {acc_avg:.4f}"
    )
    if opt.robust_train:
        summary_message += (
            f", CleanCE_avg: {epoch_summary['clean_loss']:.4f}"
            f", DegCE_avg: {epoch_summary['deg_loss']:.4f}"
            f", Cons_avg: {epoch_summary['cons_loss']:.4f}"
            f", DropHSI_avg: {epoch_summary['drop_hsi_ratio']:.4f}"
            f", DropAux_avg: {epoch_summary['drop_aux_ratio']:.4f}"
        )
    print(summary_message)
    return epoch_summary


def evaluate_epoch(data_loader, model_obj):
    if opt.max_test_batches > 0:
        limited_batches = []
        total_batches = len(data_loader)
        for i, batch in enumerate(data_loader, start=1):
            limited_batches.append(batch)
            if opt.print_freq > 0 and (i == 1 or i % opt.print_freq == 0):
                print(f"Collect test batch [{i:04d}/{total_batches:04d}]")
            if i >= opt.max_test_batches:
                print(f"Stop early after {i} test batches for smoke test.")
                break
        data_loader = limited_batches

    result_bundle = utility.evaluate_model(
        net=model_obj,
        data_loader=data_loader,
        dataset_name=opt.dataset,
        device=device,
        print_freq=max(opt.print_freq, 1),
    )
    return result_bundle["metrics"]


if __name__ == "__main__":
    print("Start train...")
    time_begin = time.time()

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        epoch_metrics = {"epoch": epoch}
        epoch_metrics.update(train_one_epoch(train_loader, model, optimizer, epoch))

        if opt.skip_test:
            print("Skip test stage for smoke test.")
        else:
            print(f"Start evaluation for {opt.dataset} with eval_split={opt.eval_split}...")
            metrics = evaluate_epoch(test_loader, model)
            epoch_metrics.update(
                {
                    "oa": metrics["oa"],
                    "aa": metrics["aa"],
                    "kappa": metrics["kappa"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            print(
                f"Eval Epoch [{epoch:03d}/{opt.epoch:03d}] "
                f"{utility.summarize_metrics(metrics)}"
            )

            utility.save_json(metric_summary_dict(metrics), last_metrics_path)
            if metrics["oa"] > best_acc:
                best_acc = metrics["oa"]
                best_epoch = epoch
                utility.save_json(metric_summary_dict(metrics), best_metrics_path)
                save_checkpoint(
                    model,
                    optimizer,
                    os.path.join(run_dirs["weights_dir"], "best.pth"),
                    epoch,
                    metrics=metrics,
                )

        save_checkpoint(
            model,
            optimizer,
            os.path.join(run_dirs["weights_dir"], "last.pth"),
            epoch,
            metrics=epoch_metrics,
        )
        utility.append_history_row(
            history_path,
            {
                "epoch": epoch,
                "train_loss": round(epoch_metrics["train_loss"], 6),
                "train_acc": round(epoch_metrics["train_acc"], 4),
                "clean_loss": round(epoch_metrics.get("clean_loss", 0.0), 6),
                "deg_loss": round(epoch_metrics.get("deg_loss", 0.0), 6),
                "cons_loss": round(epoch_metrics.get("cons_loss", 0.0), 6),
                "drop_hsi_ratio": round(epoch_metrics.get("drop_hsi_ratio", 0.0), 6),
                "drop_aux_ratio": round(epoch_metrics.get("drop_aux_ratio", 0.0), 6),
                "hsi_noise_applied_ratio": round(epoch_metrics.get("hsi_noise_applied_ratio", 0.0), 6),
                "aux_noise_applied_ratio": round(epoch_metrics.get("aux_noise_applied_ratio", 0.0), 6),
                "hsi_band_dropout_ratio": round(epoch_metrics.get("hsi_band_dropout_ratio", 0.0), 6),
                "oa": round(epoch_metrics.get("oa", 0.0), 4),
                "aa": round(epoch_metrics.get("aa", 0.0), 4),
                "kappa": round(epoch_metrics.get("kappa", 0.0), 4),
                "macro_f1": round(epoch_metrics.get("macro_f1", 0.0), 4),
                "best_oa": round(best_acc, 4),
                "best_epoch": best_epoch,
            },
        )

        elapsed = time.time() - time_begin
        print(f"Best OA: {best_acc:.4f} at epoch {best_epoch:03d}")
        print(f"Time out:{elapsed:.2f}s\n")
        logging.info("Best_acc:%.4f,Best_epoch:%03d", best_acc, best_epoch)
        logging.info("Time out:%.2fs\n", elapsed)

    utility.save_json(
        {
            "best_oa": best_acc,
            "best_epoch": best_epoch,
            "run_dir": run_dirs["run_dir"],
        },
        os.path.join(run_dirs["metrics_dir"], "run_summary.json"),
    )

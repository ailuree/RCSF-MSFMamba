import logging
import os
import time

import torch

import utility
from model.MSFMamba import Net
from setting.dataLoader import get_loader
from setting.options import opt
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
    summary = {
        "dataset": metrics["dataset"],
        "num_samples": metrics["num_samples"],
        "oa": metrics["oa"],
        "aa": metrics["aa"],
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": metrics["per_class_accuracy"],
    }
    if "gate_x_avg" in metrics:
        summary["gate_x_avg"] = metrics["gate_x_avg"]
    if "gate_y_avg" in metrics:
        summary["gate_y_avg"] = metrics["gate_y_avg"]
    if "alpha_dts_mean" in metrics:
        summary["alpha_dts_mean"] = metrics["alpha_dts_mean"]
    if "alpha_bs_mean" in metrics:
        summary["alpha_bs_mean"] = metrics["alpha_bs_mean"]
    if "alpha_cs_mean" in metrics:
        summary["alpha_cs_mean"] = metrics["alpha_cs_mean"]
    return summary


def collect_gate_means(model_obj):
    gate_x_values = []
    gate_y_values = []

    for layer in getattr(model_obj, "layers", []):
        fss_block = getattr(layer, "FSSBlock", None)
        if fss_block is None:
            continue
        gate_x = getattr(fss_block, "last_gate_x", None)
        gate_y = getattr(fss_block, "last_gate_y", None)
        if gate_x is not None:
            gate_x_values.append(gate_x.mean().item())
        if gate_y is not None:
            gate_y_values.append(gate_y.mean().item())

    gate_x_mean = sum(gate_x_values) / len(gate_x_values) if gate_x_values else 0.0
    gate_y_mean = sum(gate_y_values) / len(gate_y_values) if gate_y_values else 0.0
    return gate_x_mean, gate_y_mean


def save_checkpoint(model_obj, optimizer_obj, checkpoint_path, epoch, metrics=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model_obj.state_dict(),
        "optimizer_state_dict": optimizer_obj.state_dict(),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, checkpoint_path)


def collect_cross_state_means(model_obj):
    alpha_dts_means = []
    alpha_bs_means = []
    alpha_cs_means = []

    for layer in getattr(model_obj, "layers", []):
        fss_block = getattr(layer, "FSSBlock", None)
        if fss_block is None:
            continue
        for attention_name in ("attention1", "attention2"):
            attention = getattr(fss_block, attention_name, None)
            if attention is None:
                continue
            alpha_dts = getattr(attention, "last_alpha_dts_mean", None)
            alpha_bs = getattr(attention, "last_alpha_bs_mean", None)
            alpha_cs = getattr(attention, "last_alpha_cs_mean", None)
            if alpha_dts is not None:
                alpha_dts_means.append(alpha_dts)
            if alpha_bs is not None:
                alpha_bs_means.append(alpha_bs)
            if alpha_cs is not None:
                alpha_cs_means.append(alpha_cs)

    if not alpha_dts_means:
        return None

    return {
        "alpha_dts_mean": sum(alpha_dts_means) / len(alpha_dts_means),
        "alpha_bs_mean": sum(alpha_bs_means) / len(alpha_bs_means),
        "alpha_cs_mean": sum(alpha_cs_means) / len(alpha_cs_means),
    }


def train_one_epoch(data_loader, model_obj, optimizer_obj, epoch):
    model_obj.train()
    loss_all = 0.0
    iteration = len(data_loader)
    acc = 0.0
    num = 0
    gate_x_running = 0.0
    gate_y_running = 0.0
    alpha_dts_sum = 0.0
    alpha_bs_sum = 0.0
    alpha_cs_sum = 0.0
    alpha_count = 0

    for i, (_, xdata, hsi_pca, gt, _, _) in enumerate(data_loader, start=1):
        optimizer_obj.zero_grad()
        xdata = xdata.to(device)
        hsi_pca = hsi_pca.to(device)
        gt = gt.to(device)

        _, outputs = model_obj(hsi_pca.unsqueeze(1), xdata)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer_obj.step()

        loss_all += loss.item()
        acc += compute_accuracy(outputs, gt) * len(gt)
        num += len(gt)
        batch_gate_x, batch_gate_y = collect_gate_means(model_obj)
        gate_x_running += batch_gate_x
        gate_y_running += batch_gate_y
        cross_state_means = collect_cross_state_means(model_obj)
        if cross_state_means is not None:
            alpha_dts_sum += cross_state_means["alpha_dts_mean"]
            alpha_bs_sum += cross_state_means["alpha_bs_mean"]
            alpha_cs_sum += cross_state_means["alpha_cs_mean"]
            alpha_count += 1

        if opt.print_freq > 0 and (i == 1 or i % opt.print_freq == 0 or i == iteration):
            message = (
                f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] "
                f"Step [{i:04d}/{iteration:04d}] "
                f"Loss: {loss.item():.4f} "
                f"GateX: {batch_gate_x:.4f} "
                f"GateY: {batch_gate_y:.4f}"
            )
            if cross_state_means is not None:
                message += (
                    f" AlphaDt: {cross_state_means['alpha_dts_mean']:.4f}"
                    f" AlphaB: {cross_state_means['alpha_bs_mean']:.4f}"
                    f" AlphaC: {cross_state_means['alpha_cs_mean']:.4f}"
                )
            print(message)

        if opt.max_train_batches > 0 and i >= opt.max_train_batches:
            print(f"Stop early after {i} train batches for smoke test.")
            break

    loss_avg = loss_all / i
    acc_avg = acc / max(num, 1)
    gate_x_avg = gate_x_running / i
    gate_y_avg = gate_y_running / i
    alpha_summary = None
    if alpha_count > 0:
        alpha_summary = {
            "alpha_dts_mean": alpha_dts_sum / alpha_count,
            "alpha_bs_mean": alpha_bs_sum / alpha_count,
            "alpha_cs_mean": alpha_cs_sum / alpha_count,
        }
    logging.info(
        "Epoch [%03d/%03d], Loss_train_avg: %.4f, acc_avg: %.4f, gate_x_avg: %.4f, gate_y_avg: %.4f, alpha_dt_avg: %s, alpha_b_avg: %s, alpha_c_avg: %s",
        epoch,
        opt.epoch,
        loss_avg,
        acc_avg,
        gate_x_avg,
        gate_y_avg,
        f"{alpha_summary['alpha_dts_mean']:.4f}" if alpha_summary is not None else "N/A",
        f"{alpha_summary['alpha_bs_mean']:.4f}" if alpha_summary is not None else "N/A",
        f"{alpha_summary['alpha_cs_mean']:.4f}" if alpha_summary is not None else "N/A",
    )
    summary_message = (
        f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] Done, "
        f"Loss_avg: {loss_avg:.4f}, Acc_avg: {acc_avg:.4f}"
    )
    if alpha_summary is not None:
        summary_message += (
            f", GateX_avg: {gate_x_avg:.4f}"
            f", GateY_avg: {gate_y_avg:.4f}"
            f", AlphaDt_avg: {alpha_summary['alpha_dts_mean']:.4f}"
            f", AlphaB_avg: {alpha_summary['alpha_bs_mean']:.4f}"
            f", AlphaC_avg: {alpha_summary['alpha_cs_mean']:.4f}"
        )
    else:
        summary_message += (
            f", GateX_avg: {gate_x_avg:.4f}"
            f", GateY_avg: {gate_y_avg:.4f}"
        )
    print(summary_message)
    return loss_avg, acc_avg, gate_x_avg, gate_y_avg, alpha_summary


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
        train_loss, train_acc, gate_x_avg, gate_y_avg, alpha_summary = train_one_epoch(train_loader, model, optimizer, epoch)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
            "gate_x_avg": gate_x_avg,
            "gate_y_avg": gate_y_avg,
        }
        if alpha_summary is not None:
            epoch_metrics.update(alpha_summary)

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
                    "gate_x_avg": gate_x_avg,
                    "gate_y_avg": gate_y_avg,
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
                "oa": round(epoch_metrics.get("oa", 0.0), 4),
                "aa": round(epoch_metrics.get("aa", 0.0), 4),
                "kappa": round(epoch_metrics.get("kappa", 0.0), 4),
                "macro_f1": round(epoch_metrics.get("macro_f1", 0.0), 4),
                "gate_x_avg": round(epoch_metrics.get("gate_x_avg", 0.0), 6),
                "gate_y_avg": round(epoch_metrics.get("gate_y_avg", 0.0), 6),
                "alpha_dts_mean": round(epoch_metrics.get("alpha_dts_mean", 0.0), 6),
                "alpha_bs_mean": round(epoch_metrics.get("alpha_bs_mean", 0.0), 6),
                "alpha_cs_mean": round(epoch_metrics.get("alpha_cs_mean", 0.0), 6),
                "best_oa": round(best_acc, 4),
                "best_epoch": best_epoch,
            },
        )

        elapsed = time.time() - time_begin
        print(f"Best OA: {best_acc:.4f} at epoch {best_epoch:03d}")
        print(f"Gate means: GateX={gate_x_avg:.4f}, GateY={gate_y_avg:.4f}")
        print(f"Time out:{elapsed:.2f}s\n")
        logging.info("Best_acc:%.4f,Best_epoch:%03d", best_acc, best_epoch)
        logging.info("Gate means: GateX=%.4f, GateY=%.4f", gate_x_avg, gate_y_avg)
        logging.info("Time out:%.2fs\n", elapsed)

    utility.save_json(
        {
            "best_oa": best_acc,
            "best_epoch": best_epoch,
            "gate_x_avg": gate_x_avg,
            "gate_y_avg": gate_y_avg,
            "alpha_dts_mean": epoch_metrics.get("alpha_dts_mean", 0.0),
            "alpha_bs_mean": epoch_metrics.get("alpha_bs_mean", 0.0),
            "alpha_cs_mean": epoch_metrics.get("alpha_cs_mean", 0.0),
            "run_dir": run_dirs["run_dir"],
        },
        os.path.join(run_dirs["metrics_dir"], "run_summary.json"),
    )

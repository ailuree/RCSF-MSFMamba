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


def train_one_epoch(data_loader, model_obj, optimizer_obj, epoch):
    model_obj.train()
    loss_all = 0.0
    iteration = len(data_loader)
    acc = 0.0
    num = 0

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

        if opt.print_freq > 0 and (i == 1 or i % opt.print_freq == 0 or i == iteration):
            print(
                f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] "
                f"Step [{i:04d}/{iteration:04d}] "
                f"Loss: {loss.item():.4f}"
            )

        if opt.max_train_batches > 0 and i >= opt.max_train_batches:
            print(f"Stop early after {i} train batches for smoke test.")
            break

    loss_avg = loss_all / i
    acc_avg = acc / max(num, 1)
    logging.info(
        "Epoch [%03d/%03d], Loss_train_avg: %.4f, acc_avg: %.4f",
        epoch,
        opt.epoch,
        loss_avg,
        acc_avg,
    )
    print(f"Train Epoch [{epoch:03d}/{opt.epoch:03d}] Done, Loss_avg: {loss_avg:.4f}, Acc_avg: {acc_avg:.4f}")
    return loss_avg, acc_avg


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
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, epoch)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
        }

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

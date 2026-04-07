import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def createDatasetReport(net, data, dataset_name, device):
    class_names = DATASET_CLASS_NAMES.get(dataset_name)
    if class_names is None:
        raise ValueError(f"Unsupported dataset for reporting: {dataset_name}")
    print(f"{dataset_name} Start!")
    return createReport(net, data, class_names, device)


def createReport(net, data, class_names, device):
    global cate
    net.eval()
    labels = list(range(len(class_names)))
    total_batches = len(data) if hasattr(data, "__len__") else None
    total_samples = 0
    start_time = time.time()
    count = 0
    for batch_idx, (hsi, x, hsi_pca, test_labels,h,w) in enumerate(data, start=1):
        hsi=hsi.cuda(device)
        hsi_pca = hsi_pca.to(device)
        x = x.to(device)
        _ , outputs = net(hsi_pca.unsqueeze(1), x)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        batch_size = len(test_labels)
        total_samples += batch_size

        if total_batches is not None and (
            batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches
        ):
            elapsed = time.time() - start_time
            print(
                f"Eval Step [{batch_idx:04d}/{total_batches:04d}] "
                f"Samples: {total_samples} "
                f"Elapsed: {elapsed:.2f}s"
            )

        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))

    classification = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100

    logging.info(f'\n{classification}')
    logging.info(f'Overall accuracy (%) {oa}')
    logging.info(f'Average accuracy (%) {aa}')
    logging.info(f'Kappa accuracy (%){kappa}')
    logging.info(f'\n{confusion}')
    
    return oa,aa,kappa,each_acc

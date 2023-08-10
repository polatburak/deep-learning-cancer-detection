# Evaluation on test data

import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import importlib

from data_loading import get_train_and_test_loader


def evaluate_testset(name, model_name, params=None, labels_file=None, img_dir=None):
    """
    Evaluate a trained deep learning model on the testset.
    Args:
        name:
            Name of the output files.
        model_name:
            Module name of the model architecture.
        params:
            Special network params.
        labels_file:
            File path for image labels.
        img_dir:
            File path for images.

    """

    # Import model, e.g. "cnn_2
    model_module = importlib.import_module(f"architecture.{model_name}")

    # Create neural network
    if params:
        net = model_module.Net(img_dim=[3, 128, 128], **params)
    else:
        net = model_module.Net(img_dim=[3, 128, 128])

    # Load model parameters
    PATH = f"./{name}.pth"
    net.load_state_dict(torch.load(PATH))
    net.cpu()

    # Create data loader
    _, test_loader = get_train_and_test_loader(labels_file, img_dir)

    # Create testing metrics
    metric_collection = MetricCollection([
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
        AUROC()
    ])


    list_of_labels = []
    list_of_preds = []

    # Evaluation loop
    with torch.no_grad():
        for data in tqdm(test_loader):
            imgs, labels = data
            preds = net(imgs)
            list_of_preds.append(preds)
            list_of_labels.append(labels)


    labels = torch.cat(list_of_labels)
    preds = torch.cat(list_of_preds)

    # Calculate metrics
    metrics = metric_collection(preds, labels)


    # Create dataframe
    test_metrics = pd.DataFrame({
        'accuracy': metrics['Accuracy'].item(),
        'precision': metrics['Precision'].item(),
        'recall': metrics['Recall'].item(),
        'f1_score': metrics['F1Score'].item(),
        'auroc': metrics['AUROC'].item()
        }, index=[0])

    # Save testing metrics
    test_metrics.to_csv(f'{name}_test.csv')



if __name__=="__main__": 
    parser = ArgumentParser()
    parser.add_argument("--name", action="store", required=True)
    parser.add_argument("--model_name", action="store", required=True)
    args = parser.parse_args()
    args = vars(args)
    name = args["name"]
    model_name = args["model_name"]

    evaluate_testset(name, model_name)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import numpy as np
import torch

import os


def plot_accuracy(train_acc, test_acc, run_name, dir):
    sns.set_theme(style="whitegrid")

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_acc, label="Train Accuracy")
    sns.lineplot(x=epochs, y=test_acc, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()

    plt.tight_layout()

    save_dir = f"{os.path.dirname(__file__)}/{dir}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/accuracy_curve.png", dpi=300)


def plot_loss(loss_history, run_name, dir):
    sns.set_theme(style="whitegrid")

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=loss_history, label="Training Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")

    plt.tight_layout()

    save_dir = f"{os.path.dirname(__file__)}/{dir}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)


def plot_confusion_matrix(model, dataloader, device, run_name, label, dir):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            labels = y[label].to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="mako", square=True)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    save_dir = f"{os.path.dirname(__file__)}/{dir}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)

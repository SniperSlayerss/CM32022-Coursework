import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.metrics import confusion_matrix
import numpy as np
import torch

import os

import json
import glob


def plot_accuracy(train_acc, test_acc, run_name, dir):
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


def plot_d4(metric_type: str):
    data = {}
    for path in glob.glob(f"D4/results/{metric_type}/*.json"):
        with open(path) as f:
            d = json.load(f)
        margin = d["margin"]
        label = d["label_key"]  # "fine" or "coarse"
        data[(label, margin)] = d

    margins = [0.3, 0.5, 1.0, 1.5]
    ks = [5, 10, 50, 100]
    k_labels = ["R@5", "R@10", "R@50", "R@100"]

    COLOR = {0.3: "#378ADD", 0.5: "#1D9E75", 1.0: "#BA7517", 1.5: "#E24B4A"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distance Weighted triplet sampling", fontsize=13, fontweight="normal", y=1.01)

    ax = axes[0]
    for m in margins:
        print(m)
        for label_key, ls in [("fine", "-"), ("coarse", "--")]:
            entry = data.get((label_key, m))
            if entry is None:
                continue
            recalls = [
                entry["recall_at_5"],
                entry["recall_at_10"],
                entry["recall_at_50"],
                entry["recall_at_100"],
            ]
            recalls_pct = [round(r * 100, 2) for r in recalls]
            print(label_key)

            print(recalls_pct)
            lw = 2.0 if label_key == "fine" else 1.4
            ax.plot(k_labels, recalls_pct, color=COLOR[m], linestyle=ls, linewidth=lw, marker="o", markersize=4)

    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_title("Recall@k by margin and label type", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="#cccccc", linewidth=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    margin_handles = [mlines.Line2D([], [], color=COLOR[m], linewidth=2, label=f"m={m}") for m in margins]
    style_handles = [
        mlines.Line2D([], [], color="gray", linewidth=2, linestyle="-", label="Fine"),
        mlines.Line2D([], [], color="gray", linewidth=1.4, linestyle="--", label="Coarse"),
    ]
    ax.legend(handles=margin_handles + style_handles, fontsize=9, ncol=2, frameon=False, loc="upper left")

    ax2 = axes[1]

    configs = []
    for label_key in ["fine", "coarse"]:
        for m in margins:
            entry = data.get((label_key, m))
            if entry is None:
                continue
            configs.append(
                {
                    "name": f"{'F' if label_key == 'fine' else 'C'} m={m}",
                    "pos": entry["avg_pos_dist"],
                    "neg": entry["avg_neg_dist"],
                    "ratio": entry["pos_neg_ratio"],
                    "color": COLOR[m],
                    "hatch": "" if label_key == "fine" else "///",
                }
            )

    x = np.arange(len(configs))
    bar_w = 0.35

    for i, cfg in enumerate(configs):
        ax2.bar(x[i] - bar_w / 2, cfg["pos"], width=bar_w, color=cfg["color"], alpha=0.8, hatch=cfg["hatch"], label="_nolegend_")
        ax2.bar(x[i] + bar_w / 2, cfg["neg"], width=bar_w, color=cfg["color"], alpha=0.3, hatch=cfg["hatch"], label="_nolegend_")

    ax2.set_xticks(x)
    ax2.set_xticklabels([c["name"] for c in configs], rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Avg Euclidean distance", fontsize=11)
    ax2.set_title("Positive vs negative distance separation", fontsize=11)
    ax2.tick_params(labelsize=10)
    ax2.grid(axis="y", color="#cccccc", linewidth=0.5, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    pos_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.8, label="Avg pos dist")
    neg_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.3, label="Avg neg dist")
    fine_patch = plt.Rectangle((0, 0), 1, 1, fc="silver", label="Fine (solid)")
    coarse_patch = plt.Rectangle((0, 0), 1, 1, fc="silver", hatch="///", label="Coarse (hatched)")
    ax2.legend(handles=[pos_patch, neg_patch, fine_patch, coarse_patch], fontsize=9, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig("triplet_results.png", dpi=150, bbox_inches="tight")
    print("Saved triplet_results.png")
    plt.show()


if __name__ == "__main__":
    plot_d4("distance")

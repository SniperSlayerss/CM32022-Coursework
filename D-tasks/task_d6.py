import matplotlib.pyplot as plt
import numpy as np
import task_d2
import task_d4
import torch
from dataset import CIFAR_100_MEAN, CIFAR_100_STD, init_dataloaders
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE

COARSE_CLASSES = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]


def extract_embeddings(model, dataloader, device: str):
    model.eval()

    embeddings = []
    labels = []
    images = []

    mean = torch.tensor(CIFAR_100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_100_STD).view(1, 3, 1, 1)

    with torch.no_grad():
        for test_images, test_targets in dataloader:
            test_images = test_images.to(device)
            emb = model(test_images)
            embeddings.append(emb.cpu())
            labels.append(test_targets["coarse"])

            # De-normalise images
            imgs = test_images.cpu() * std + mean
            imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).astype(np.uint8)

            images.append(imgs)

    return (
        torch.cat(embeddings).numpy(),
        torch.cat(labels).numpy(),
        np.concatenate(images),
    )


def plot_tsne_images(
    proj: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(proj[:, 0].min() - 5, proj[:, 0].max() + 5)
    ax.set_ylim(proj[:, 1].min() - 5, proj[:, 1].max() + 5)
    ax.set_facecolor("#f5f5f5")

    for idx in range(len(proj)):
        img = images[idx]
        oi = OffsetImage(img, zoom=0.55)
        oi.image.axes = ax
        ab = AnnotationBbox(
            oi,
            (proj[idx, 0], proj[idx, 1]),
            frameon=False,
            pad=0.0,
        )
        ax.add_artist(ab)

    # Bold label at the centroid of each coarse class
    for class_idx in range(20):
        mask = labels == class_idx
        if mask.sum() == 0:
            continue
        cx = proj[mask, 0].mean()
        cy = proj[mask, 1].mean()
        ax.text(
            cx,
            cy,
            COARSE_CLASSES[class_idx],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color="lime",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
        )

    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_tsne(proj: np.ndarray, labels: np.ndarray, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps["tab20"]

    for class_idx in range(len(COARSE_CLASSES)):
        mask = labels == class_idx
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=3,
            alpha=0.6,
            color=cmap(class_idx / 20),
            label=COARSE_CLASSES[class_idx],
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(
        loc="upper right",
        markerscale=3,
        fontsize=6,
        ncol=2,
        framealpha=0.7,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = init_dataloaders()

    print("Starting D2 t-SNE...")
    d2_model = task_d2.prepare_test()
    d2_model.to(device)

    d2_emb, d2_coarse_labels, d2_images = extract_embeddings(d2_model.backbone, data.test_dataloader, device)

    d2_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(d2_emb)

    plot_tsne(
        d2_proj,
        d2_coarse_labels,
        title="t-SNE of D2 model embeddings",
        save_path="./output/d6_tsne_d2.png",
    )

    plot_tsne_images(
        d2_proj,
        d2_images,
        d2_coarse_labels,
        title="t-SNE of D2 model embeddings (images)",
        save_path="./output/d6_tsne_d2_images.png",
    )
    print("Finished D2 t-SNE!")

    for margin in [0.3, 0.5, 1.0]:
        print(f"Starting D4 t-SNE (margin={margin})...")
        d4_model = task_d4.prepare_test(margin=margin, fine_labels=False)
        d4_model.to(device)

        d4_emb, d4_coarse_labels, d4_images = extract_embeddings(d4_model, data.test_dataloader, device)

        d4_proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(d4_emb)

        plot_tsne(
            d4_proj,
            d4_coarse_labels,
            title=f"t-SNE of D4 model embeddings (margin={margin})",
            save_path=f"./output/d6_tsne_d4_m={margin}_coarse.png",
        )

        plot_tsne_images(
            d4_proj,
            d4_images,
            d4_coarse_labels,
            title=f"t-SNE of D4 model embeddings (margin={margin}, images)",
            save_path=f"./output/d6_tsne_d4_m={margin}_coarse_images.png",
        )

        print(f"Finished D4 t-SNE (margin={margin})!")


if __name__ == "__main__":
    main()

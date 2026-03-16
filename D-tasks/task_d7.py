import os

import numpy as np
import task_d1
import task_d2
import torch
from dataset import init_dataloaders


def prepare_test():
    # TODO: Load the model from task D1 and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply leave the code below as is. Otherwise, please adjust.

    model = task_d1.prepare_test()
    return model.backbone


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_emb: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in dataloader:
            emb = model(images.to(device))
            emb = emb.view(emb.size(0), -1).cpu()
            all_emb.append(emb)
            all_labels.append(targets["coarse"])

    return torch.cat(all_emb, dim=0), torch.cat(all_labels, dim=0)


def run_knn(
    model: torch.nn.Module,
    data,
    device: str,
) -> dict[int, float]:
    k_vals = [1, 3, 5, 10, 20]

    train_emb, train_coarse = extract_embeddings(model, data.train_dataloader, device)
    test_emb, test_coarse = extract_embeddings(model, data.test_dataloader, device)

    distances = torch.cdist(test_emb, train_emb)

    results: dict[int, float] = dict()

    knn_indices = distances.topk(max(k_vals), largest=False).indices
    knn_coarse = train_coarse[knn_indices]

    print("\nD1 coarse k-NN results:")
    for k in k_vals:
        preds = knn_coarse[:, :k].mode(dim=1).values
        acc = preds.eq(test_coarse).float().mean().item()
        results[k] = acc
        print(f"{k=}  {acc * 100:>9.2f}%")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = init_dataloaders()

    print("Starting D1 k-NN...")
    d1_model = prepare_test().to(device)
    d1_results = run_knn(d1_model, data, device)
    print("Finished D1 k-NN!")

    print("Starting D2 k-NN...")
    d2_model = task_d2.prepare_test().to(device)
    d2_acc, _, _ = task_d2.evaluate_model(d2_model, data.test_dataloader, device)
    print(f"D2 accuracy: {d2_acc * 100:.2f}%")

    print("Finished D2 k-NN!")

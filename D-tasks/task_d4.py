from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbone import new_backbone
from dataset import init_dataloaders
import os
import json


class TripletModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        z = self.backbone(x)
        # Add L2 normalisation to output embeddings
        # Makes Euclidean Distance behave like cosine similarity
        return torch.nn.functional.normalize(z, p=2, dim=1, eps=1e-8)


# Original triplet loss before adding batch hard mining
# Pushing positives closer and negatives further by at least margin
def triplet_loss(anchor, positive, negative, margin=0.5):
    # Euclidean distance
    d_pos = torch.nn.functional.pairwise_distance(anchor, positive)
    d_neg = torch.nn.functional.pairwise_distance(anchor, negative)
    loss = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return loss.mean()


# Trains using the most 'informative' triplets, where the loss is higher
def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    dists = torch.cdist(embeddings, embeddings)

    labels = labels.unsqueeze(1)
    # Mask where labels share same class
    # This means each row i, pos_mask[i] gives which columns are valid pos for anchor i
    pos_mask = labels == labels.T
    # Mask where labels do not share same class
    neg_mask = labels != labels.T

    # Remove self-comparisons from positives
    # Otherwise each image would be its own hardest positive
    diag_mask = torch.eye(len(embeddings), device=embeddings.device).bool()
    pos_mask[diag_mask] = False

    # Find anchors that have at least one valid positive and one negative
    valid_anchors = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid_anchors.any():
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)  # If none than return 0

    dists = dists[valid_anchors]
    pos_mask = pos_mask[valid_anchors]
    neg_mask = neg_mask[valid_anchors]

    # Hardest positive for each anchor, i.e. same class image that is furthest away
    hardest_positive = (dists * pos_mask).max(dim=1)[0]

    # Hardest negative for each anchor, i.e different class image that is closest
    max_dist = dists.detach().max()
    masked_neg_dists = dists.clone()
    masked_neg_dists[~neg_mask] = max_dist + 1
    hardest_negative = masked_neg_dists.min(dim=1)[0]

    loss = torch.clamp(hardest_positive - hardest_negative + margin, min=0.0)
    return loss.mean()


def train_model(model, train_dataloader, test_dataloader, device, label_key="fine", margin=0.5):
    num_epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_recall = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_dataloader:
            x = x.to(device)
            labels = y[label_key].to(device)

            optimizer.zero_grad()

            embeddings = model(x)

            loss = batch_hard_triplet_loss(embeddings, labels, margin)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        recall, avg_pos_dist, avg_neg_dist = evaluate_model(model, test_dataloader, device, label_key)

        # Save model when recall@5 improves
        if recall[5] > best_recall:
            best_recall = recall[5]

            dir_path = f"{os.path.dirname(__file__)}/models"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(model.state_dict(), f"{dir_path}/d4_m={margin}_{label_key}.pth")

        scheduler.step()

        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_dataloader):.4f}, Recall@5={recall[5]:.4f}, pos_dist={avg_pos_dist:.4f}, neg_dist={avg_neg_dist:.4f}")


# Rank images by embedding distance,
# then check how many of the top k are in the same class,
# then dividing by total class size excluding itself
def recall_at_k(embeddings, labels, dists, k):
    recalls = []
    for i in range(len(embeddings)):
        sorted_idx = dists[i].argsort()[:k]
        retrieved_labels = labels[sorted_idx]
        class_size = (labels == labels[i]).sum().item() - 1  # exclude self
        hits = (retrieved_labels == labels[i]).sum().item()
        recalls.append(hits / class_size if class_size > 0 else 0)
    return sum(recalls) / len(recalls)


def evaluate_model(model, dataloader, device, label_key="fine"):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            emb = model(x.to(device))
            all_embeddings.append(emb.cpu())
            all_labels.append(y[label_key])

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    dists = torch.cdist(embeddings, embeddings)
    dists.fill_diagonal_(float("inf"))

    recalls = {}
    for k in [5, 10, 50, 100]:
        recalls[k] = recall_at_k(embeddings, labels, dists, k)

    # average anchor-positive and anchor-negative distances
    pos_dists, neg_dists = [], []
    dists_with_self = torch.cdist(embeddings, embeddings)
    for i in range(len(embeddings)):
        pos_mask = labels == labels[i]
        pos_mask[i] = False  # exclude self
        neg_mask = labels != labels[i]
        pos_dists.append(dists_with_self[i][pos_mask].mean().item())
        neg_dists.append(dists_with_self[i][neg_mask].mean().item())

    return (
        recalls,
        sum(pos_dists) / len(pos_dists),
        sum(neg_dists) / len(neg_dists),
    )


# TODO Save your best models and store them at './models/d4_m={margin}_fine.pth' or ./models/d4_m={margin}_coarse.pth,
#  depending on whether you trained the model with triplets formed with the fine or coarse labels.
#  {margin} is the margin value that you used to train the model. You must upload at least two models, one for the
#  fine-grained version and one for the coarse-grained version, specifying the margin value. You can upload multiple
#  models trained with different margin values
def prepare_test(margin, fine_labels):
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 576), where B is the batch size and 576 is the
    #  embedding dimension. Make sure that the correct model is loaded depending on the margin and fine_labels parameters
    #  where `margin` is a float and `fine_labels` is a boolean that if True/False will load the model trained with triplets
    #  formed with the fine/coarse labels.

    backbone = new_backbone()
    model = TripletModel(backbone)

    # do not edit from here downwards
    s = "fine" if fine_labels else "coarse"
    weights_path = f"models/d4_m={margin}_{s}.pth"

    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


if __name__ == "__main__":
    # backbone_before = copy.deepcopy(model.backbone.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    save_dir = f"{os.path.dirname(__file__)}/D4/results"
    os.makedirs(save_dir, exist_ok=True)

    # def train_model(model, train_dataloader, test_dataloader, device, label_key="fine", margin=0.5
    for margin in [0.3, 0.5, 1.0]:
        for label_key in ["fine", "coarse"]:
            batch_size = 64
            if label_key == "fine":
                batch_size = 128

            data = init_dataloaders(batch_size=batch_size)
            print(f"{label_key} : {margin} : bs {batch_size}")
            f = new_backbone()
            model = TripletModel(f).to(device)
            train_model(model, data.train_dataloader, data.test_dataloader, device, label_key, margin)

            model.load_state_dict(
                torch.load(
                    f"{os.path.dirname(__file__)}/models/d4_m={margin}_{label_key}.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
            recall, avg_pos_dist, avg_neg_dist = evaluate_model(model, data.test_dataloader, device, label_key)

            print(
                "----------- Test set -----------\n",
                f"Recall@5:   {recall[5]:.4f}\n",
                f"Recall@10:  {recall[10]:.4f}\n",
                f"Recall@50:  {recall[50]:.4f}\n",
                f"Recall@100: {recall[100]:.4f}\n",
                f"Avg pos dist:  {avg_pos_dist:.4f}\n",
                f"Avg neg dist:  {avg_neg_dist:.4f}\n",
                f"Pos/neg ratio: {avg_pos_dist / avg_neg_dist:.4f}\n",
            )

            summary = {
                "label_key": label_key,
                "margin": margin,
                "recall_at_5": recall[5],
                "recall_at_10": recall[10],
                "recall_at_50": recall[50],
                "recall_at_100": recall[100],
                "avg_pos_dist": avg_pos_dist,
                "avg_neg_dist": avg_neg_dist,
                "pos_neg_ratio": avg_pos_dist / avg_neg_dist,
                "bs": batch_size,
            }

            with open(f"{save_dir}/m={margin}_{label_key}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

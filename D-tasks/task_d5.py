import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from backbone import new_backbone

from d5_dataset import init_dataloaders
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt



class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = backbone

    
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.model(out)
        return out

class EpisodeSampler:
    def __init__(self, dataset, N=5, P=10, Q=20):
        self.N = N
        self.P = P
        self.Q = Q

        self.dataset = dataset
        self.classes = defaultdict(list)

        for idx, label in enumerate(self.dataset.fine_labels):
            self.classes[int(label)].append(idx)

        self.class_names = [
            cls for cls, idxs in self.classes.items()
            if len(idxs) >= self.P + self.Q
        ]


    def sample_data(self):
        selected_classes = random.sample(self.class_names, self.N)
    
        support_indices = []
        query_indicies = []

        for cls in selected_classes:
            indices = self.classes[cls]
            samples = random.sample(indices, self.P + self.Q)

            support_indices.extend(samples[:self.P])
            query_indicies.extend(samples[self.P:])

        return support_indices, query_indicies
    
    
    def compute_prototypes(self, support_emb):

        support_emb = support_emb.view((self.N, self.P, -1))

        prototypes = support_emb.mean(dim=1)

        return prototypes
    
    def fetch_batch(self, indices, device):
        xs = []
        ys = []

        for idx in indices:
            x, y = self.dataset[idx]
            xs.append(x)
            ys.append(y["fine"])
         
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, dtype=torch.long, device=device)

        return x, y




def train_model(
        model: Model, 
        train_dataset: Dataset, 
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader,  
        support_dataloader: DataLoader | None, 
        k: int, 
        device: str
        ):
    
    num_epochs = 2
    num_episodes = 100
    N, P, Q = 5, 10, 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_recall = 0.0

    sampler = EpisodeSampler(train_dataset, N=N, P=P, Q=Q)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for episode in range(num_episodes):
            optimizer.zero_grad()

            support_indices, query_indices = sampler.sample_data()

            support_x, _ = sampler.fetch_batch(support_indices, device)
            query_x, _ = sampler.fetch_batch(query_indices, device)
          
            support_emb = model(support_x)
            query_emb = model(query_x)

            prototypes = sampler.compute_prototypes(support_emb)

            distances = torch.cdist(query_emb, prototypes)

            local_targets = torch.arange(N, device=device).repeat_interleave(Q)

            loss = criterion(-distances, local_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc, _, recall = evaluate_model(model, test_dataloader, support_dataloader, k, device)
     
        # Save model when recall@5 improves
        if recall[5] > best_recall:
            best_recall = recall[5]

            dir_path = f"{os.path.dirname(__file__)}/models"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(model.state_dict(), f"{dir_path}/d5.pth")

        scheduler.step()

        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_dataloader):.4f}, Recall@5={recall[5]:.4f},  Test={test_acc * 100:.2f}%")


def get_support_embeddings(model: Model, support_dataloader: DataLoader | None, device: str):
    model.eval()
    support_embeddings = []
    support_targets = []

    with torch.no_grad():
        for data, target in support_dataloader:
            data, target = data.to(device), target["fine"].to(device)
            emb = model(data)
            support_embeddings.append(emb)
            support_targets.append(target)

    return torch.cat(support_embeddings, dim=0), torch.cat(support_targets, dim=0)


def compute_knn_accuracy(distances, query_targets, support_targets, K, exclude_self=False):
    nearest_neighbors = torch.topk(distances, k=K, dim=1, largest=False).indices
    nearest_targets = support_targets[nearest_neighbors]
    predicted_targets = torch.mode(nearest_targets, dim=1).values
    acc = (predicted_targets == query_targets).float().mean().item()
    return acc, predicted_targets


def recall_at_k(query_labels, support_labels, dists, k):
    recalls = []
    for i in range(len(query_labels)):
        sorted_idx = dists[i].argsort()[:k]
        retrieved_labels = support_labels[sorted_idx]
        class_size = (support_labels == query_labels[i]).sum().item()
        hits = (retrieved_labels == query_labels[i]).sum().item()
        recalls.append(hits / class_size if class_size > 0 else 0)
    return sum(recalls) / len(recalls)


def evaluate_model(model: Model, test_dataloader: DataLoader, support_dataloader: DataLoader, k: int, device):
    model.eval()
    y_pred = []
    # y_true = []
    t_acc = 0

    all_test_embeddings = []
    all_test_targets = []


    with torch.no_grad():
        support_embeddings, support_targets = get_support_embeddings(model, support_dataloader, device)

        for data, batch_targets in test_dataloader:
            data = data.to(device)
            batch_targets = batch_targets["fine"].to(device)

            batch_test_embeddings = model(data)

            all_test_embeddings.append(batch_test_embeddings)
            all_test_targets.append(batch_targets)

            distances = torch.cdist(batch_test_embeddings, support_embeddings)

            acc, predicted_targets = compute_knn_accuracy(
                distances, batch_targets, support_targets, k
            )

            t_acc += acc
            
        
            y_pred.append(predicted_targets)
            # y_true.append(batch_targets)

    test_embeddings = torch.cat(all_test_embeddings, dim=0)
    test_targets = torch.cat(all_test_targets, dim=0)
    distances = torch.cdist(test_embeddings, support_embeddings)

 
    recalls = {}
    for k in [5, 10, 50, 100]:
        recalls[k] = recall_at_k(test_targets, support_targets, distances, k)

    y_pred = torch.cat(y_pred)

    # y_true = torch.cat(y_true)
    # acc = (y_pred == y_true).float().mean().item()
    
    acc = t_acc / len(test_dataloader)

    return acc, y_pred, recalls


# TODO save your best model and store it at './models/d5.pth'

def prepare_test():
    # TODO: Load the model and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply return model.backbone

    backbone = new_backbone()
    model = Model(backbone)

    # do not edit from here downwards
    weights_path = 'models/d5.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


if __name__ == "__main__":

    support = True

    # KNN parameter
    K = 5

    # Choice of support set
    support_set = 5


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # data = init_dataloaders()
    data = init_dataloaders("/zero_shot", support)

    f = new_backbone()
    # TODO do we train the backbone?
    # f.eval()
    # for param in f.parameters():
    #     param.requires_grad = False
    model = Model(f).to(device)

    support_dataloader = {
        1: data.support1_dataloader,
        5: data.support5_dataloader,
        10: data.support10_dataloader
    }


    train_model(model, data.train_set, data.train_dataloader, data.test_dataloader, support_dataloader[support_set], 5, device)
    
    model.load_state_dict(
        torch.load(
            f"{os.path.dirname(__file__)}/models/d5.pth",
            weights_only=True,
            map_location=device,
        )
    )


    acc, y_pred , recall  = evaluate_model(model, data.test_dataloader, data.support10_dataloader, 5, device)

    print(
        "\n----------- Test set -----------",
        f"\nAccuracy: {100.0 * acc:.0f}%\n",
        f"\nRecall@5={recall[5]:.4f}"
    )

    # backbone_after = model.backbone.state_dict()
    # Check if weights are unchanged
    # for key in backbone_before:
    #     if not torch.equal(backbone_before[key], backbone_after[key]):
    #         print(f"Weight {key} changed!")



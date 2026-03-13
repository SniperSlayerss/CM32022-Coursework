import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbone import new_backbone
from dataset import init_dataloaders
import os
import json
from dataclasses import dataclass


class ModelBig(nn.Module):
    def __init__(self, backbone, num_classes, dropout=[0.2, 0.1]):
        super(ModelBig, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out


class ModelSmall(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.2):
        super(ModelSmall, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out


class ModelNarrow(nn.Module):
    def __init__(self, backbone, num_classes, dropout=[0.2, 0.1]):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ModelWide(nn.Module):
    def __init__(self, backbone, num_classes, dropout=[0.2, 0.1, 0.1]):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 2048),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout[2]),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


@dataclass
class RunConfig:
    name: str
    model_type: str  # big, small, narrow, wide
    dropout: list | float  # [d1, d2] for big, float for small
    optim_type: str  # adam or sgd
    lr: float
    weight_decay: float
    scheduler_type: str  # cosine, warm_restarts, or none
    label_smoothing: float
    batch_size: int = 32
    num_epochs: int = 200
    note: str = ""


def build_training_objects(config: RunConfig, device: str):
    backbone = new_backbone()
    if config.model_type == "big":
        model = ModelBig(backbone, 20, dropout=config.dropout).to(device)
    elif config.model_type == "small":
        model = ModelSmall(backbone, 20, dropout=config.dropout).to(device)
    elif config.model_type == "narrow":
        model = ModelNarrow(backbone, 20, dropout=config.dropout).to(device)
    else:
        model = ModelWide(backbone, 20, dropout=config.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    if config.optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optim_type}")

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler_type == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    elif config.scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler_type}")

    return model, criterion, optimizer, scheduler


def train_model(model, criterion, optimizer, scheduler, config: RunConfig, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str):
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    lr_history = []

    best_acc = 0.0
    dir_path = f"{os.path.dirname(__file__)}/models"
    os.makedirs(dir_path, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y["coarse"].to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = predictions.argmax(dim=1)
            train_correct += pred.eq(y).sum().item()
            train_total += y.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc, test_loss, top3_acc = evaluate_model(model, test_dataloader, device)
        avg_loss = total_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{dir_path}/d2.pth")

        if scheduler is not None:
            scheduler.step()

        print(f"[{config.name}] Epoch {epoch + 1}/{config.num_epochs}: Loss={avg_loss:.4f}, Train={train_acc:.2f}%, Test={test_acc * 100:.2f}%, Top3={top3_acc * 100:.2f}%, LR={current_lr:.6f}")

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc * 100)
        train_loss_history.append(avg_loss)
        lr_history.append(current_lr)

    return train_acc_history, test_acc_history, train_loss_history, lr_history


def evaluate_model(model: ModelBig | ModelSmall, test_dataloader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    total_correct = 0
    top3_correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target["coarse"].to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

            top3_pred = output.topk(3, dim=1).indices
            top3_correct += sum(target[i] in top3_pred[i] for i in range(len(target)))

    n = len(test_dataloader.dataset)
    return total_correct / n, test_loss / n, top3_correct / n


def save_summary(config: RunConfig, train_acc, test_acc, loss_history, lr_history, top3_acc):
    gap_history = [tr - te for tr, te in zip(train_acc, test_acc)]
    best_idx = test_acc.index(max(test_acc))

    summary = {
        "run": config.name,
        "note": config.note,
        "model_type": config.model_type,
        "dropout": config.dropout,
        "optim": config.optim_type,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "scheduler": config.scheduler_type,
        "label_smoothing": config.label_smoothing,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        # Results
        "best_test_acc": max(test_acc),
        "best_epoch": best_idx + 1,
        "best_gap": gap_history[best_idx],
        "final_train_acc": train_acc[-1],
        "final_test_acc": test_acc[-1],
        "final_gap": gap_history[-1],
        "top3_acc": top3_acc * 100,
        "min_loss": min(loss_history),
    }

    save_dir = f"{os.path.dirname(__file__)}/D2/{config.name}"
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{config.name} summary:")
    print(f"Best test acc : {max(test_acc):.2f}% (epoch {best_idx + 1})")
    print(f"Top-3 acc     : {top3_acc * 100:.2f}%")
    print(f"Gap at best   : {gap_history[best_idx]:.2f}%")


# TODO save your best model and store it at './models/d2.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 20), where B is the batch size
    #  and 20 is the number of classes. The output is the prediction of your classifier, providing a score for each
    #  class, for each image in input

    # TODO CHANGE TO CORRECT MODEL
    model = ModelSmall(new_backbone(), 20, dropout=0.5).to("cuda" if torch.cuda.is_available() else "cpu")

    # do not edit from here downwards
    weights_path = "models/d2.pth"
    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


RUNS = [
    RunConfig(name="rbest2", model_type="small", dropout=0.5, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="warm_restarts", label_smoothing=0.2, batch_size=32, note="best"),
    # # Scheduler
    # RunConfig(name="r02_warm_restarts", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="warm_restarts", label_smoothing=0.1, batch_size=32, note="scheduler: warm restarts"),
    # RunConfig(name="r03_no_scheduler", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="none", label_smoothing=0.1, batch_size=32, note="scheduler: none"),
    # # Optimiser
    # RunConfig(name="r04_sgd", model_type="big", dropout=[0.2, 0.1], optim_type="sgd", lr=0.05, weight_decay=5e-4, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="SGD + Nesterov"),
    # # Learning rate
    # RunConfig(name="r05_lr_low", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-4, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="lr: 1e-4"),
    # RunConfig(name="r06_lr_high", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=5e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="lr: 5e-3"),
    # # Batch size
    # RunConfig(name="r07_bs64", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=64, note="batch size 64"),
    # RunConfig(name="r08_bs128", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=128, note="batch size 128"),
    # # Dropout
    # RunConfig(name="r09_dropout_low", model_type="big", dropout=[0.1, 0.05], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="dropout: very low"),
    # RunConfig(name="r10_dropout_mid", model_type="big", dropout=[0.3, 0.2], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="dropout: medium"),
    # RunConfig(name="r11_dropout_high", model_type="big", dropout=[0.5, 0.4], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="dropout: high"),
    # # Label smoothing
    # RunConfig(name="r12_smooth_0", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.0, batch_size=32, note="no label smoothing"),
    # RunConfig(name="r13_smooth_02", model_type="big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.2, batch_size=32, note="label smoothing 0.2"),
    # # Architecture
    # RunConfig(name="r14_small", model_type="small", dropout=0.2, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="small: 576->512->20"),
    # RunConfig(name="r15_small", model_type="small", dropout=0.5, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="small: 576->512->100"),
    # RunConfig(name="r16_narrow", model_type="narrow", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="narrow: 576->256->20"),
    # RunConfig(name="r17_narrow", model_type="narrow", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="narrow: 576->256->100"),
    # RunConfig(name="r18_wide", model_type="wide", dropout=[0.2, 0.1, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="wide: 576->2048->1024->512->20"),
    # RunConfig(name="r19_wide", model_type="wide", dropout=[0.3, 0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, batch_size=32, note="wide: 576->2048->1024->512->100"),
]

if __name__ == "__main__":
    import plot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for config in RUNS:
        print(f"Starting {config.name}: {config.note}")

        data = init_dataloaders(batch_size=config.batch_size)
        model, criterion, optimizer, scheduler = build_training_objects(config, device)

        train_acc, test_acc, loss_history, lr_history = train_model(
            model,
            criterion,
            optimizer,
            scheduler,
            config,
            data.train_dataloader,
            data.test_dataloader,
            device,
        )

        model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/models/d2.pth", weights_only=True, map_location=device))
        _, _, top3_acc = evaluate_model(model, data.test_dataloader, device)

        dir = "D2"
        plot.plot_accuracy(train_acc, test_acc, config.name, dir)
        plot.plot_loss(loss_history, config.name, dir)
        plot.plot_confusion_matrix(model, data.test_dataloader, device, config.name, "coarse", dir)

        save_summary(config, train_acc, test_acc, loss_history, lr_history, top3_acc)

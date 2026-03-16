import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbone import new_backbone
from dataset import init_dataloaders
import os
import json
from dataclasses import dataclass


class ModelShared(nn.Module):
    def __init__(self, backbone, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fine_head = nn.Linear(512, 100)
        self.coarse_head = nn.Linear(512, 20)

    def forward(self, x):
        out = self.backbone(x)
        out = self.shared(out)
        return self.fine_head(out), self.coarse_head(out)


class ModelSeparate(nn.Module):
    def __init__(self, backbone, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.fine_classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 100),
        )
        self.coarse_classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 20),
        )

    def forward(self, x):
        out = self.backbone(x)
        return self.fine_classifier(out), self.coarse_classifier(out)


class ModelSharedBig(nn.Module):
    def __init__(self, backbone, dropout=[0.3, 0.2]):
        super().__init__()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
        )
        self.fine_head = nn.Linear(512, 100)
        self.coarse_head = nn.Linear(512, 20)

    def forward(self, x):
        out = self.backbone(x)
        out = self.shared(out)
        return self.fine_head(out), self.coarse_head(out)


@dataclass
class RunConfig:
    name: str
    model_type: str  # "shared", "shared_big", "separate"
    dropout: list | float
    optim_type: str  # "adam" or "sgd"
    lr: float
    weight_decay: float
    scheduler_type: str  # "cosine", "warm_restarts", "none"
    label_smoothing: float
    fine_weight: float
    coarse_weight: float
    batch_size: int = 32
    num_epochs: int = 200
    note: str = ""


def build_training_objects(config: RunConfig, device: str):
    backbone = new_backbone()

    if config.model_type == "shared":
        model = ModelShared(backbone, dropout=config.dropout).to(device)
    elif config.model_type == "shared_big":
        model = ModelSharedBig(backbone, dropout=config.dropout).to(device)
    elif config.model_type == "separate":
        model = ModelSeparate(backbone, dropout=config.dropout).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    fine_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    coarse_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

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
        raise ValueError(f"Unknown optim_type: {config.optim_type}")

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.scheduler_type == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    elif config.scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")

    return model, fine_criterion, coarse_criterion, optimizer, scheduler


def train_model(model, fine_criterion, coarse_criterion, optimizer, scheduler, config: RunConfig, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str):
    train_fine_acc_history = []
    train_coarse_acc_history = []
    train_both_acc_history = []
    test_fine_acc_history = []
    test_coarse_acc_history = []
    test_both_acc_history = []
    loss_history = []
    lr_history = []

    best_avg_acc = 0.0
    os.makedirs(f"{os.path.dirname(__file__)}/models", exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        fine_correct = coarse_correct = both_correct = total = 0

        for x, y in train_dataloader:
            x = x.to(device)
            y_fine = y["fine"].to(device)
            y_coarse = y["coarse"].to(device)

            optimizer.zero_grad()
            pred_fine, pred_coarse = model(x)

            loss = config.fine_weight * fine_criterion(pred_fine, y_fine) + config.coarse_weight * coarse_criterion(pred_coarse, y_coarse)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pf = pred_fine.argmax(dim=1)
            pc = pred_coarse.argmax(dim=1)
            fine_correct += pf.eq(y_fine).sum().item()
            coarse_correct += pc.eq(y_coarse).sum().item()
            both_correct += (pf.eq(y_fine) & pc.eq(y_coarse)).sum().item()
            total += y_fine.size(0)

        train_fine_acc = 100.0 * fine_correct / total
        train_coarse_acc = 100.0 * coarse_correct / total
        train_both_acc = 100.0 * both_correct / total

        test_fine_acc, test_coarse_acc, test_both_acc, _ = evaluate_model(model, test_dataloader, device)
        avg_acc = (test_fine_acc + test_coarse_acc) / 2
        avg_loss = total_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), f"{os.path.dirname(__file__)}/models/d3.pth")

        if scheduler is not None:
            scheduler.step()

        print(
            f"[{config.name}] Epoch {epoch + 1}/{config.num_epochs}: "
            f"Loss={avg_loss:.4f} LR={current_lr:.6f}\n"
            f"  Train  fine={train_fine_acc:.2f}%  coarse={train_coarse_acc:.2f}%  both={train_both_acc:.2f}%\n"
            f"  Test   fine={test_fine_acc * 100:.2f}%  coarse={test_coarse_acc * 100:.2f}%  both={test_both_acc * 100:.2f}%"
        )

        train_fine_acc_history.append(train_fine_acc)
        train_coarse_acc_history.append(train_coarse_acc)
        train_both_acc_history.append(train_both_acc)
        test_fine_acc_history.append(test_fine_acc * 100)
        test_coarse_acc_history.append(test_coarse_acc * 100)
        test_both_acc_history.append(test_both_acc * 100)
        loss_history.append(avg_loss)
        lr_history.append(current_lr)

    return (train_fine_acc_history, train_coarse_acc_history, train_both_acc_history, test_fine_acc_history, test_coarse_acc_history, test_both_acc_history, loss_history, lr_history)


def evaluate_model(model, test_dataloader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    correct_fine = correct_coarse = correct_both = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            y_fine = target["fine"].to(device)
            y_coarse = target["coarse"].to(device)

            pred_fine, pred_coarse = model(data)
            test_loss += nn.functional.cross_entropy(pred_fine, y_fine, reduction="sum").item()
            test_loss += nn.functional.cross_entropy(pred_coarse, y_coarse, reduction="sum").item()

            pf = pred_fine.argmax(dim=1)
            pc = pred_coarse.argmax(dim=1)
            correct_fine += pf.eq(y_fine).sum().item()
            correct_coarse += pc.eq(y_coarse).sum().item()
            correct_both += (pf.eq(y_fine) & pc.eq(y_coarse)).sum().item()

    n = len(test_dataloader.dataset)
    return correct_fine / n, correct_coarse / n, correct_both / n, test_loss / n


def save_summary(config: RunConfig, results: tuple, top5_fine: float, top5_coarse: float, top3_fine: float, top3_coarse: float):
    os.makedirs("results", exist_ok=True)
    train_fine, train_coarse, train_both, test_fine, test_coarse, test_both, loss_history, lr_history = results

    # Use avg for picking best epoch
    avg_test = [(f + c) / 2 for f, c in zip(test_fine, test_coarse)]
    best_idx = avg_test.index(max(avg_test))

    fine_gap = [tr - te for tr, te in zip(train_fine, test_fine)]
    coarse_gap = [tr - te for tr, te in zip(train_coarse, test_coarse)]

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
        "fine_weight": config.fine_weight,
        "coarse_weight": config.coarse_weight,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        # Results
        "best_epoch": best_idx + 1,
        "best_fine_acc": max(test_fine),
        "best_coarse_acc": max(test_coarse),
        "best_both_acc": max(test_both),
        "best_avg_acc": max(avg_test),
        "fine_gap_at_best": fine_gap[best_idx],
        "coarse_gap_at_best": coarse_gap[best_idx],
        "final_fine_acc": test_fine[-1],
        "final_coarse_acc": test_coarse[-1],
        "top5_fine_acc": top5_fine * 100,
        "top3_fine_acc": top3_fine * 100,
        "top5_coarse_acc": top5_coarse * 100,
        "top3_coarse_acc": top3_coarse * 100,
    }

    save_dir = f"{os.path.dirname(__file__)}/D3/{config.name}"
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{config.name} summary:")
    print(f"  Best fine acc   : {max(test_fine):.2f}%  (epoch {best_idx + 1})")
    print(f"  Best coarse acc : {max(test_coarse):.2f}%")
    print(f"  Best both acc   : {max(test_both):.2f}%")
    print(f"  Top-5 fine      : {top5_fine * 100:.2f}%")


def evaluate_topN_fine(model, test_dataloader: DataLoader, device: str, n):
    model.eval()
    topn_correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            y_fine = target["fine"].to(device)
            pred_fine, _ = model(data)
            topn_pred = pred_fine.topk(n, dim=1).indices
            topn_correct += sum(y_fine[i] in topn_pred[i] for i in range(len(y_fine)))
    return topn_correct / len(test_dataloader.dataset)


def evaluate_topN_coarse(model, test_dataloader: DataLoader, device: str, n):
    model.eval()
    topn_correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            y_coarse = target["coarse"].to(device)
            _, pred_coarse = model(data)
            topn_pred = pred_coarse.topk(n, dim=1).indices
            topn_correct += sum(y_coarse[i] in topn_pred[i] for i in range(len(y_coarse)))
    return topn_correct / len(test_dataloader.dataset)


# TODO save your best model and store it at './models/d3.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    # model = ModelSeparate(new_backbone(), dropout=0.4).to("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelSeparate(new_backbone(), dropout=0.4)

    # do not edit from here downwards
    weights_path = "models/d3.pth"
    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


RUNS = [
    RunConfig(name="rbest2", model_type="separate", dropout=0.4, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.2, fine_weight=1.0, coarse_weight=1.0, note="best"),
    # Scheduler
    # RunConfig(name="r01_cosine", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="baseline"),
    # RunConfig(name="r02_warm_restarts", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="warm_restarts", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="scheduler: warm restarts"),
    # RunConfig(name="r03_no_scheduler", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="none", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="scheduler: none"),
    # # Optimiser
    # RunConfig(name="r04_sgd", model_type="shared", dropout=0.3, optim_type="sgd", lr=0.05, weight_decay=5e-4, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="SGD + Nesterov"),
    # # Learning rate
    # RunConfig(name="r05_lr_low", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-4, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="lr: 1e-4"),
    # RunConfig(name="r06_lr_high", model_type="shared", dropout=0.3, optim_type="adam", lr=5e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="lr: 5e-3"),
    # # Batch size
    # RunConfig(name="r07_bs64", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, batch_size=64, note="batch size 64"),
    # RunConfig(name="r08_bs128", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, batch_size=128, note="batch size 128"),
    # # Dropout
    # RunConfig(name="r09_dropout_low", model_type="shared", dropout=0.1, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="dropout: 0.1"),
    # RunConfig(name="r10_dropout_high", model_type="shared", dropout=0.5, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="dropout: 0.5"),
    # # Label smoothing
    # RunConfig(name="r11_smooth_0", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.0, fine_weight=1.0, coarse_weight=1.0, note="no label smoothing"),
    # RunConfig(name="r12_smooth_02", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.2, fine_weight=1.0, coarse_weight=1.0, note="label smoothing 0.2"),
    # # Loss weighting
    # RunConfig(name="r13_fine_heavy", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=2.0, coarse_weight=1.0, note="fine loss weight x2"),
    # RunConfig(name="r14_coarse_heavy", model_type="shared", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=2.0, note="coarse loss weight x2"),
    # # Architecture
    # RunConfig(name="r15_shared_big", model_type="shared_big", dropout=[0.3, 0.2], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="wider shared trunk"),
    # RunConfig(name="r16_shared_big", model_type="shared_big", dropout=[0.2, 0.1], optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="wider shared trunk"),
    # RunConfig(name="r17_separate", model_type="separate", dropout=0.2, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="separate heads, no shared trunk"),
    # RunConfig(name="r18_separate", model_type="separate", dropout=0.3, optim_type="adam", lr=1e-3, weight_decay=1e-3, scheduler_type="cosine", label_smoothing=0.1, fine_weight=1.0, coarse_weight=1.0, note="separate heads, no shared trunk"),
]


if __name__ == "__main__":
    import plot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for config in RUNS:
        print(f"Starting {config.name}: {config.note}")

        data = init_dataloaders(batch_size=config.batch_size)
        model, fine_criterion, coarse_criterion, optimizer, scheduler = build_training_objects(config, device)

        results = train_model(
            model,
            fine_criterion,
            coarse_criterion,
            optimizer,
            scheduler,
            config,
            data.train_dataloader,
            data.test_dataloader,
            device,
        )

        model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/models/d3.pth", weights_only=True, map_location=device))
        top5_fine = evaluate_topN_fine(model, data.test_dataloader, device, 5)
        top3_fine = evaluate_topN_fine(model, data.test_dataloader, device, 3)
        top5_coarse = evaluate_topN_coarse(model, data.test_dataloader, device, 5)
        top3_coarse = evaluate_topN_coarse(model, data.test_dataloader, device, 3)

        dir = "D3"
        train_fine, train_coarse, train_both, test_fine, test_coarse, test_both, loss_h, _ = results
        plot.plot_accuracy(train_fine, test_fine, f"{config.name}_fine", dir)
        plot.plot_accuracy(train_coarse, test_coarse, f"{config.name}_coarse", dir)
        plot.plot_loss(loss_h, config.name, dir)
        # plot.plot_confusion_matrix(model, data.test_dataloader, device, config.name, "fine", dir)

        save_summary(config, results, top5_fine, top5_coarse, top3_fine, top3_coarse)

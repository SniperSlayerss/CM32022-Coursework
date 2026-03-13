import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbone import new_backbone

from dataset import init_dataloaders
import os


class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fine_classifier = nn.Linear(512, 100)
        self.coarse_classifier = nn.Linear(512, 20)

    def forward(self, x):
        out = self.backbone(x)
        out = self.shared(out)
        return self.fine_classifier(out), self.coarse_classifier(out)


def train_model(
    model: Model, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str
):
    num_epochs = 200

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        train_fine_correct = 0
        train_coarse_correct = 0
        train_both_correct = 0

        train_fine_total = 0
        train_coarse_total = 0

        for x, y in train_dataloader:
            x = x.to(device)
            y_fine = y["fine"].to(device)
            y_coarse = y["coarse"].to(device)

            optimizer.zero_grad()
            pred_fine, pred_coarse = model(x)
            loss = nn.CrossEntropyLoss()(pred_fine, y_fine) + nn.CrossEntropyLoss()(
                pred_coarse, y_coarse
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            fine = pred_fine.argmax(dim=1)
            coarse = pred_coarse.argmax(dim=1)
            train_fine_correct += fine.eq(y_fine).sum().item()
            train_coarse_correct += coarse.eq(y_coarse).sum().item()
            train_both_correct += (fine.eq(y_fine) & coarse.eq(y_coarse)).sum().item()

            train_fine_total += y_fine.size(0)
            train_coarse_total += y_coarse.size(0)

        train_fine_acc = 100.0 * train_fine_correct / train_fine_total
        train_coarse_acc = 100.0 * train_coarse_correct / train_coarse_total
        train_both_acc = 100.0 * train_both_correct / train_fine_total
        test_fine_acc, test_coarse_acc, test_both_acc, _ = evaluate_model(
            model, test_dataloader, device
        )

        avg_acc = (test_fine_acc + test_coarse_acc) / 2
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(
                model.state_dict(),
                f"{os.path.dirname(__file__)}/models/d3.pth",
            )
        scheduler.step()
        print(
            f"Epoch {epoch + 1}: Loss={total_loss / len(train_dataloader):.4f},\n"
            f"    Train Fine Acc={train_fine_acc:.2f}%, Train Coarse Acc={train_coarse_acc:.2f}%, Train Both Acc={train_both_acc:.2f}%\n"
            f"    Test Fine Acc={100.0 * test_fine_acc:.2f}%, Test Coarse Acc={100.0 * test_coarse_acc:.2f}%, Test Both Acc={100.0 * test_both_acc:.2f}%"
        )


def evaluate_model(model: Model, test_dataloader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    correct_fine = 0
    correct_coarse = 0
    correct_both = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            y_fine = target["fine"].to(device)
            y_coarse = target["coarse"].to(device)
            pred_fine, pred_coarse = model(data)
            test_loss += nn.functional.cross_entropy(
                pred_fine, y_fine, reduction="sum"
            ).item()
            test_loss += nn.functional.cross_entropy(
                pred_coarse, y_coarse, reduction="sum"
            ).item()

            correct_fine += (
                pred_fine.argmax(dim=1, keepdim=True)
                .eq(y_fine.view_as(pred_fine.argmax(dim=1, keepdim=True)))
                .sum()
                .item()
            )
            correct_coarse += (
                pred_coarse.argmax(dim=1, keepdim=True)
                .eq(y_coarse.view_as(pred_coarse.argmax(dim=1, keepdim=True)))
                .sum()
                .item()
            )
            correct_both += (
                (
                    pred_fine.argmax(dim=1).eq(y_fine)
                    & pred_coarse.argmax(dim=1).eq(y_coarse)
                )
                .sum()
                .item()
            )
        test_loss /= len(test_dataloader.dataset)

    fine_acc = correct_fine / len(test_dataloader.dataset)
    coarse_acc = correct_coarse / len(test_dataloader.dataset)
    both_acc = correct_both / len(test_dataloader.dataset)
    return fine_acc, coarse_acc, both_acc, test_loss


# TODO save your best model and store it at './models/d3.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    model = Model(new_backbone()).to("cuda" if torch.cuda.is_available() else "cpu")

    # do not edit from here downwards
    weights_path = "models/d3.pth"
    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=map_location)
    )

    return model


if __name__ == "__main__":
    # backbone_before = copy.deepcopy(model.backbone.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    data = init_dataloaders()

    f = new_backbone()
    # TODO do we train the backbone?
    # f.eval()
    # for param in f.parameters():
    #     param.requires_grad = False
    model = Model(f).to(device)

    train_model(model, data.train_dataloader, data.test_dataloader, device)
    model.load_state_dict(
        torch.load(
            f"{os.path.dirname(__file__)}/models/d3.pth",
            weights_only=True,
            map_location=device,
        )
    )
    acc_fine, acc_coarse, acc_both, loss = evaluate_model(
        model, data.test_dataloader, device
    )

    print(
        "----------- Test set -----------\n",
        f"Average loss: {loss:.4f},\n",
        f"Accuracy fine: {100.0 * acc_fine:.0f}%\n",
        f"Accuracy coarse: {100.0 * acc_coarse:.0f}%\n",
        f"Accuracy both: {100.0 * acc_both:.0f}%\n",
    )

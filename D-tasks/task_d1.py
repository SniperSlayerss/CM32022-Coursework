import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backbone import new_backbone

from dataset import init_dataloaders
import os

# import copy


# https://docs.pytorch.org/examples/
class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Model, self).__init__()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x)
        # print(out.shape)
        out = self.classifier(out)
        return out


def train_model(
    model: Model, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str
):
    num_epochs = 200

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(
    #     [
    #         {"params": model.backbone.parameters(), "lr": 1e-2},
    #         {"params": model.classifier.parameters(), "lr": 1e-2},
    #     ],
    #     momentum=0.9,
    #     weight_decay=1e-3,
    #     nesterov=True,
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=[1e-3, 1e-2],
    #     steps_per_epoch=len(train_dataloader),
    #     epochs=num_epochs
    # )

    best_acc = 0.0
    for epoch in range(num_epochs):
        # model.classifier.train()
        model.train()
        total_loss = 0.0

        train_correct = 0
        train_total = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y["fine"].to(device)

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
        test_acc, _ = evaluate_model(model, test_dataloader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                model.state_dict(),
                f"{os.path.dirname(__file__)}/models/best_model1.pth",
            )

        scheduler.step()
        print(
            f"Epoch {epoch + 1}: Loss={total_loss / len(train_dataloader):.4f}, Train Acc={train_acc:.2f}%, Test Acc={100.0 * test_acc:.2f}%"
        )


def evaluate_model(model: Model, test_dataloader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    number_of_correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target["fine"].to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            number_of_correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_dataloader.dataset)

    acc = number_of_correct / len(test_dataloader.dataset)
    return acc, test_loss


# TODO save your best model and store it at './models/d1.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input

    model = Model(new_backbone(), 100)

    # do not edit from here downwards
    weights_path = "models/d1.pth"
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
    model = Model(f, 100).to(device)

    train_model(model, data.train_dataloader, data.test_dataloader, device)
    model.load_state_dict(
        torch.load(
            f"{os.path.dirname(__file__)}/models/best_model1.pth",
            weights_only=True,
            map_location=device,
        )
    )
    acc, loss = evaluate_model(model, data.test_dataloader, device)

    print(
        "\n----------- Test set -----------",
        f"\nAverage loss: {loss:.4f},",
        f"\nAccuracy: {100.0 * acc:.0f}%\n",
    )

    # backbone_after = model.backbone.state_dict()
    # Check if weights are unchanged
    # for key in backbone_before:
    #     if not torch.equal(backbone_before[key], backbone_after[key]):
    #         print(f"Weight {key} changed!")

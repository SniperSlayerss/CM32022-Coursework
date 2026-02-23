import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backbone import new_backbone

from dataset import init_dataloaders

# import copy

# https://docs.pytorch.org/examples/
class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Model, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Linear(in_features=576, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        out = self.backbone(x)
        # print(out.shape)
        out = self.classifier(out)
        return out


def train_model(model: Model, train_dataloader: DataLoader, device: str):
    # Train
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        # model.classifier.train()
        model.train()
        total_loss = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y["coarse"].to(device)

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}"
        )


def evaluate_model(
    model: Model, test_dataloader: DataLoader, test_set: Dataset, device: str
):
    # Test
    model.eval()
    test_loss = 0
    number_of_correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target["coarse"].to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            number_of_correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_set)

    print(
        "\n----------- Test set -----------",
        f"\nAverage loss: {test_loss:.4f},",
        f"\nAccuracy: {number_of_correct}/{len(test_set)} ({100.0 * number_of_correct / len(test_set):.0f}%)\n",
    )


# TODO save your best model and store it at './models/d2.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 20), where B is the batch size
    #  and 20 is the number of classes. The output is the prediction of your classifier, providing a score for each
    #  class, for each image in input

    model = Model(new_backbone(), 20) # TODO change this to your model

    # do not edit from here downwards
    weights_path = "models/d2.pth"
    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=map_location)
    )

    return model

if __name__ == "__main__":
    # backbone_before = copy.deepcopy(model.backbone.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = init_dataloaders()

    f = new_backbone()
    # TODO do we train the backbone?
    # f.eval()
    # for param in f.parameters():
    #     param.requires_grad = False
    model = Model(f, 20).to(device)

    train_model(model, data.train_dataloader, device)
    evaluate_model(model, data.test_dataloader, data.test_set, device)

    # backbone_after = model.backbone.state_dict()
    # Check if weights are unchanged
    # for key in backbone_before:
    #     if not torch.equal(backbone_before[key], backbone_after[key]):
    #         print(f"Weight {key} changed!")

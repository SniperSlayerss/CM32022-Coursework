import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backbone import new_backbone
from dataset import init_dataloaders


class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Linear(576, 256),
            nn.ReLU(),
        )
        self.fine_classifier = nn.Linear(256, 100)
        self.coarse_classifier = nn.Linear(256, 20)

    def forward(self, x):
        out = self.backbone(x)
        out = self.shared(out)
        return self.fine_classifier(out), self.coarse_classifier(out)


def train_model(model: Model, train_dataloader: DataLoader, device: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y_fine = y["fine"].to(device)
            y_coarse = y["coarse"].to(device)

            optimizer.zero_grad()
            pred_fine, pred_coarse = model(x)
            loss = nn.CrossEntropyLoss()(pred_fine, y_fine) + nn.CrossEntropyLoss()(pred_coarse, y_coarse)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")


def evaluate_model(model: Model, test_dataloader: DataLoader, test_set: Dataset, device: str):
    model.eval()
    correct_fine = correct_coarse = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y_fine = y["fine"].to(device)
            y_coarse = y["coarse"].to(device)

            pred_fine, pred_coarse = model(x)
            correct_fine += pred_fine.argmax(dim=1).eq(y_fine).sum().item()
            correct_coarse += pred_coarse.argmax(dim=1).eq(y_coarse).sum().item()

    n = len(test_set)
    print(f"\n----------- Test set -----------")
    print(f"Fine accuracy:   {correct_fine}/{n} ({100.0 * correct_fine / n:.1f}%)")
    print(f"Coarse accuracy: {correct_coarse}/{n} ({100.0 * correct_coarse / n:.1f}%)")

# TODO save your best model and store it at './models/d3.pth'
def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    model = None  # TODO change this to your model

    # do not edit from here downwards
    weights_path = 'models/d3.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = init_dataloaders()
    model = Model(new_backbone()).to(device)
    train_model(model, data.train_dataloader, device)
    evaluate_model(model, data.test_dataloader, data.test_set, device)
    # torch.save(model.state_dict(), 'models/d3.pth')

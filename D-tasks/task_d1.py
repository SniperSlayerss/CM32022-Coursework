import torch
import torch.nn as nn
import os
import dataset

# https://docs.pytorch.org/examples/
# TODO save your best model and store it at './models/d1.pth'
class Model(nn.Module):
    def __init__(self, num_of_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f1 = nn.Linear(in_features=1600, out_features=128)
        self.relu1 = nn.ReLU()
        self.f2 = nn.Linear(in_features=128, out_features=num_of_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_p1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_p2(out)

        out = out.reshape(out.size(0), -1)
        out = self.f1(out)
        out = self.relu1(out)
        out = self.f2(out)
        return out


def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input

    model = Model(100)

    # do not edit from here downwards
    weights_path = "models/d1.pth"
    print(f"Loading weights from {weights_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=map_location)
    )

    return model


if __name__ == "__main__":
    training_set = dataset.CIFAR100(
        f"{os.path.dirname(__file__)}/data/cifar-100-python", train=True
    )
    test_set = dataset.CIFAR100(
        f"{os.path.dirname(__file__)}/data/cifar-100-python", train=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        training_set, batch_size=64, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}"
        )

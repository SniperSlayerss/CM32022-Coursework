import torch

# TODO save your best model and store it at './models/d1.pth'


def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input

    model = None  # TODO change this to your model

    # do not edit from here downwards
    weights_path = 'models/d1.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model



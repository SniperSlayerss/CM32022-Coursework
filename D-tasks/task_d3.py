import torch

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

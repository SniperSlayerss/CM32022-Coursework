import torch

# TODO save your best model and store it at './models/d5.pth'

def prepare_test():
    # TODO: Load the model and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply return model.backbone

    model = None # TODO change this to your model

    # do not edit from here downwards
    weights_path = 'models/d5.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model
import torch

# TODO Save your best models and store them at './models/d4_m={margin}_fine.pth' or ./models/d4_m={margin}_coarse.pth,
#  depending on whether you trained the model with triplets formed with the fine or coarse labels.
#  {margin} is the margin value that you used to train the model. You must upload at least two models, one for the
#  fine-grained version and one for the coarse-grained version, specifying the margin value. You can upload multiple
#  models trained with different margin values


def prepare_test(margin, fine_labels):
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 576), where B is the batch size and 576 is the
    #  embedding dimension. Make sure that the correct model is loaded depending on the margin and fine_labels parameters
    #  where `margin` is a float and `fine_labels` is a boolean that if True/False will load the model trained with triplets
    #  formed with the fine/coarse labels.

    model = None  # TODO change this to your model

    # do not edit from here downwards
    s = 'fine' if fine_labels else 'coarse'
    weights_path = f'models/d4_m={margin}_{s}.pth'

    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


import task_d1

def prepare_test():
    # TODO: Load the model from task D1 and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply leave the code below as is. Otherwise, please adjust.

    model = task_d1.prepare_test()
    return model.backbone

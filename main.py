# # main file

# roadmap
# 1.1 from here you can launch the trained algorithms and visualize the results
# NB algorithms are trained and saved on their own modules
# 1.2: exporting a usable criterion for outliers
# 1.3: training of new models on the go with function train_model ...
# 1.4: interface ?

# imports
import torch
from visu import visu


def get_data(params: dict):
    # get the dataloader (PyTorch object) from data.py
    # params : test/train/validation sets and how to slice them ; random=42 etc, batches of number, lengths ...
    # TODO implement it !
    # TODO try and except error to see if data is adequate
    # NB should be independant from the datatype ? like if we use timeseries later
    # the only change should be to data.py
    raise NotImplementedError("get_data")


def get_model(params: dict):
    '''
    :name: nomenclature of model ex: AE_leo-17-0001.pth
    '''
    # get the saved model from saved_models (file in .pth
    # refer to README.md for nomenclature of the name

    # TODO try and except error to see if model works with the data format
    model = torch.load("saved_models\{}")
    return model


def visualize(params: dict, dataloader, model):
    # use the choice of visualisation for the model(s) selected and the data selected as true/false
    # params dico : model name for plotting, what metrics to compute, plot mcmc etc ?
    # TODO : implement for 1 model
    # TODO : for more than 1 models
    # TODO : getting out a criterion function automatically + uncertainty on it based on results
    token = visu(params)
    # TODO check results
    # TODO visu returns a criterion (boolean classifier) to use
    return token


def main(params: dict):
    # sequence of action
    loader = get_data(params)
    print("data loaded")
    model = get_model(params)
    print("model loaded")
    results = visualize(params, dataloader, model)
    print("all done")
    return results


if __name__ == "__main__":
    # construct dict with console
    # what to put in dict ?
    # refer to each function in main
    params = {}
    model_name = input("model_name")
    params["dataset"] = "test"  # "train" for training
    params["model_name"] = model_name
    main(params)

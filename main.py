# # main file

# roadmap
# 1.1 from here you can launch the trained algorithms and visualize the results
# NB algorithms are trained and saved on their own modules
# 1.2: exporting a usable criterion for outliers
# 1.3: training of new models on the go with function train_model ...
# 1.4: interface ?


def get_data(params):
    # get the dataloader (PyTorch object) from data.py
    # params : test/train/validation sets and how to slice them ; random=42 etc, batches of number, lengths ...
    # TODO implement it !
    # TODO try and except error to see if data is adequate
    # NB should be independant from the datatype ? like if we use timeseries later
    # the only change should be to data.py
    raise NotImplementedError("get_data")


def get_model(params):
    # get the saved model from models.MODEL_XX.py module$
    # TODO implement it !
    # TODO try and except error to see if model works with the data format
    from models.algo_example import load
    raise NotImplementedError("get_model")


def visualize(params, model):
    # use the choice of visualisation for the model(s) selected and the data selected as true/false
    # TODO : implement for 1 model & check results
    # TODO : for more than 1 models
    # TODO : getting out a criterion function automatically + uncertainty on it based on results
    raise NotImplementedError("visualize")


def main(params):
    # sequence of action
    raise NotImplementedError("main")


if __name__ == "__main__":
    main()

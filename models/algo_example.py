# a typical module template fro training and loading a model


def load():
    # loading model in main.py with torch.load()
    # return a functional model : please test
    raise NotImplementedError("please define the loading function")


if __name__ == "__main__":
    # training
    # use data.py to do so (or not ...)

    # saving

    # torch func to save& load
    # torch.save(model, "saved_models/NAME.pth")
    # torch.load("saved_models/NAME.pth")

    raise NotImplementedError("please train your model")

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
from models.AE_thomasB import Autoencoder

# from models.AE_remi import Net


def get_data(params: dict):
    # get the dataloader (PyTorch object) from data.py
    # params : test/train/validation sets and how to slice them ; random=42 etc, batches of number, lengths ...
    # TODO implement it !
    # TODO try and except error to see if data is adequate
    # NB should be independant from the datatype ? like if we use timeseries later
    # the only change should be to data.py

    # ajouter un paramètre pour savoir si on veut train ou test
    from data import data
    return data(params)


def get_models(params: dict):
    '''
    :name: nomenclature of model ex: AE_leo-17-0001.pth
    '''
    # for roc, get the saved model from saved_models (file in .pth
    # for tab, get the 10 saved models with each digit as an outlier
    # refer to README.md for nomenclature of the name

    # TODO try and except error to see if model works with the data format
    model_name = params['model_name'].split('-')[0]
    print(params['model_name'], model_name)
    if model_name == 'NAE_remi':
        exec('from models.{} import Net'.format(model_name))
    else:
        print('from models.{} import Autoencoder'.format(model_name))
        exec('from models.{} import Autoencoder'.format(model_name))

    if params['visu_choice'] == "roc":
        model = torch.load("saved_models/{}".format(params['model_name']))
        return [model]
    if params['visu_choice'] == "tab":
        models = []
        for i in range(10):
            try:
                model = torch.load("saved_models\{}".format(
                    model_name+"-"+str(i)+"-"+params['model_index']+'.pth'))
                models.append(model)
            except:
                print('Fail to load '+str(i))
        return models


def visualize(params: dict, dataloader, models):
    # use the choice of visualisation for the model(s) selected and the data selected as true/false
    # params dico : model name for plotting, what metrics to compute, plot mcmc etc ?
    # TODO : implement for 1 model
    # TODO : for more than 1 models
    # TODO : getting out a criterion function automatically + uncertainty on it based on results
    token = visu(params, dataloader, models)
    # TODO check results
    # TODO visu returns a criterion (boolean classifier) to use
    return token


def main(params: dict):
    # sequence of action
    loader = get_data(params)
    print("data loaded")
    models = get_models(params)
    print("model loaded")
    results = visualize(params, loader, models)
    print("all done")
    return results


if __name__ == "__main__":
    # construct dict with console
    # what to put in dict ?
    # refer to each function in main
    params = {}  # TODO a file to make sure we all have the same ?
    visu_choice = input("choice of visualisation : roc or tab")
    params["dataset"] = "test"  # "train" for training
    params['visu_choice'] = visu_choice

    if visu_choice == "roc":
        model_name = input("model_name")
        liste_outliers = input('outliers').split(',')
        for i in range(len(liste_outliers)):
            liste_outliers[i] = int(liste_outliers[i])
        params['outliers'] = liste_outliers
    if visu_choice == "tab":
        model_name = input('models_name')
        model_index = input('models_index')
        params["model_index"] = model_index
    params["model_name"] = model_name

    main(params)

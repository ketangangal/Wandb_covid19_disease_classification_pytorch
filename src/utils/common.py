import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content

def get_unique_filename(filename, typ):
    if typ == 'Plot':
        unique_filename = time.strftime(f"{filename}._%Y_%m_%d_%H_%M.png")
        return unique_filename
    elif typ == 'Model':
        unique_filename = time.strftime(f"{filename}._%Y_%m_%d_%H_%M.pt")
        return unique_filename
    else:
        return None

def save_plot(loss, acc, name, path):
    unique_name1 = get_unique_filename(name, typ='Plot')
    path_to_plot1 = os.path.join(path, unique_name1)
    fig = pd.DataFrame(data={'Loss': loss, 'Accuracy': acc}).plot()
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.savefig(path_to_plot1)


def save_model(model, name, path):
    unique_name = get_unique_filename(name, typ='Model')
    path_to_model = os.path.join(path, unique_name)
    torch.save(model, path_to_model)


from src.model_structure.model import NeuralNetwork
from src.custom_dataloader.customdata import CustomDataset
from src.custom_exception.exception import CustomException
from src.utils.common import read_config, save_model, save_plot
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import wandb
import time
import os


wandb.init(project="Image_Classification", entity="playbook-k10")
def train_net(num_epochs=None, Batch_size=None, learning_rate=None, eps=None, shuffle=None, in_channels=None,output=None):
    try:
        # Creating DataLoader
        train_DataLoader = DataLoader(CustomDataset('train'), batch_size=Batch_size, shuffle=shuffle)

        # Initialization of the neuralNetwork
        net = NeuralNetwork(in_channels, output)
        net.to(device)

        # Initialization Of optimizer and Cost Functions
        optimizer = Adam(net.parameters(), lr=learning_rate, eps=float(eps))
        cost = CrossEntropyLoss()
        wandb.watch(net,optimizer,log="all")
        losses = []
        accuracy = []

        for epoch in range(num_epochs):
            for batch, data in enumerate(train_DataLoader):
                images = data[0].to(device)
                labels = data[1].to(device)
                optimizer.zero_grad()

                # Prediction and Loss Calculation
                predictions = net.forward(images)
                loss = cost(predictions, labels)

                # Backward propagation
                loss.backward()

                # Weight adjustments
                optimizer.step()

                losses.append(loss.item())
                # acc = (correct / total)  < -- > correct is (predictions == labels)
                acc = torch.mean((torch.argmax(predictions, 1) == labels).float()).item()
                accuracy.append(acc)
                wandb.log({"loss": losses[-1], "Accuracy": accuracy[-1]})
            print(f" Epoch: {epoch}     Loss: {losses[-1]}     Accuracy: {accuracy[-1]}")
        path = os.path.join('.', 'training_result', str(time.strftime("Model._%Y_%m_%d_%H_%M")))
        if not os.path.exists(path):
            os.mkdir(path)
            save_plot(losses, accuracy, 'plot', path)
            save_model(net, 'model', path)

        print("Model and plot Saved")

    except Exception as e:
        raise e


if __name__ == '__main__':
    config = read_config('./config.yaml')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training Using Device : {device}')
    train_net(num_epochs=config['parameters']['Num_Epochs'],
              Batch_size=config['parameters']['Batch_size'],
              learning_rate=config['parameters']['learning_rate'],
              eps=config['parameters']['eps'],
              shuffle=config['parameters']['shuffle'],
              in_channels=config['parameters']['in_channels'],
              output=config['parameters']['output'])

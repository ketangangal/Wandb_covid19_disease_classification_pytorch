from src.model_structure.model import NeuralNetwork
from src.custom_dataloader.customdata import CustomDataset
from src.custom_exception.exception import CustomException
from src.custom_logger.logger import CustomLogger
from src.utils.common import read_config, save_model, save_plot
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from from_root import from_root
import sys
import torch
import wandb
import time
import os

run = wandb.init(job_type="experiment-1", project="Image_Classification", entity="playbook-k10", name="train-ex-1")
logger = CustomLogger("logs")
print(run.name)


class ModelTrainer:
    def __init__(self, config: dict):
        self.num_epochs = config['train_parameters']['Num_Epochs']
        self.Batch_size = config['train_parameters']['Batch_size']
        self.learning_rate = config['train_parameters']['learning_rate']
        self.eps = config['train_parameters']['eps']
        self.shuffle = config['train_parameters']['shuffle']
        self.in_channels = config['train_parameters']['in_channels']
        self.output = config['train_parameters']['output']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.losses = []
        self.accuracy = []
        logger.info(f'Training Using Device : {self.device}')

    def __configuration(self):
        try:
            train_DataLoader = DataLoader(CustomDataset('train'), batch_size=self.Batch_size, shuffle=self.shuffle)
            net = NeuralNetwork(self.in_channels, self.output).to(self.device)
            cost = CrossEntropyLoss()
            optimizer = Adam(net.parameters(), lr=self.learning_rate, eps=float(self.eps))
            logger.info("Created Train Loader, Net, Cost, Optimizer")
            return train_DataLoader, net, cost, optimizer
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)

    def __save_artifacts(self, net):
        try:
            path = os.path.join('.', 'training_result', str(time.strftime("Model._%Y_%m_%d_%H_%M")))
            if not os.path.exists(path):
                os.mkdir(path)
                save_plot(self.losses, self.accuracy, 'plot', path)
                save_model(net, 'model', path)
            logger.info(f"Saved Artifacts At {path}")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)

    def train_net(self):
        try:
            train_DataLoader, net, cost, optimizer = self.__configuration()
            for epoch in range(self.num_epochs):
                for batch, data in enumerate(train_DataLoader):
                    images = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    optimizer.zero_grad()

                    predictions = net.forward(images)
                    loss = cost(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    self.losses.append(loss.item())
                    # acc = (correct / total)  < -- > correct is (predictions == labels)
                    acc = torch.mean((torch.argmax(predictions, 1) == labels).float()).item()
                    self.accuracy.append(acc)
                    wandb.log({"train" : {"loss": self.losses[-1], "Accuracy": self.accuracy[-1]}})
                print(f" Epoch: {epoch}     Loss: {self.losses[-1]}     Accuracy: {self.accuracy[-1]}")
                logger.info(f" Epoch: {epoch}     Loss: {self.losses[-1]}     Accuracy: {self.accuracy[-1]}")

            self.__save_artifacts(net)

            print("Model and plot Saved")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)


if __name__ == '__main__':
    config_path = os.path.join(from_root(), 'src', 'configurations', 'config.yaml')
    config = read_config(config_path)
    trainer = ModelTrainer(config)
    trainer.train_net()

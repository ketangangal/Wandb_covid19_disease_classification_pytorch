from src.custom_dataloader.customdata import CustomDataset
from src.custom_exception.exception import CustomException
from src.custom_logger.logger import CustomLogger
from src.utils.common import read_config
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from from_root import from_root
import sys
import torch
import wandb
import os

wandb.init(job_type="experiment-1", project="Image_Classification", entity="playbook-k10", name="test-ex-1")
logger = CustomLogger("logs")


class ModelTester:
    def __init__(self, config):
        self.num_epochs = config['test_parameters']['Num_Epochs']
        self.Batch_size = config['test_parameters']['Batch_size']
        self.learning_rate = config['test_parameters']['learning_rate']
        self.eps = config['test_parameters']['eps']
        self.shuffle = config['test_parameters']['shuffle']
        self.in_channels = config['test_parameters']['in_channels']
        self.output = config['test_parameters']['output']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_loss = 0
        self.test_accuracy = 0
        self.total = 0
        self.total_batch = 0

    def configuration(self):
        try:
            test_DataLoader = DataLoader(CustomDataset('test'), batch_size=self.Batch_size, shuffle=self.shuffle)
            net = torch.load("./training_result/Model._2022_05_05_12_28/model._2022_05_05_12_28.pt").to(self.device)
            cost = CrossEntropyLoss()
            optimizer = Adam(net.parameters(), lr=self.learning_rate, eps=float(self.eps))
            net.eval()
            return test_DataLoader, net, cost, optimizer
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)

    def test_net(self):
        try:
            test_DataLoader, net, cost, optimizer = self.configuration()
            with torch.no_grad():
                holder = []
                for batch, data in enumerate(test_DataLoader):
                    images = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    output = net(images)
                    loss = cost(output, labels)

                    predictions = torch.argmax(output, 1)

                    for i in zip(images, labels, predictions):
                        h = list(i)
                        h[0] = wandb.Image(h[0])
                        holder.append(h)

                    print(f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}", )

                    self.test_loss += loss.item()
                    self.test_accuracy += (predictions == labels).sum().item()
                    self.total_batch += 1
                    self.total += labels.size(0)

            print(
                f"Model  -->   Loss : {self.test_loss / self.total_batch} Accuracy : {(self.test_accuracy / self.total) * 100} %")
            wandb.log({"Test Loss": self.test_loss / self.total_batch,
                       "Test Accuracy": (self.test_accuracy / self.total) * 100})
            my_table = wandb.Table(data=holder, columns=["image", "label", "class_prediction"])
            wandb.log({"Predictions": my_table})

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)


if __name__ == '__main__':
    config_path = os.path.join(from_root(), 'src', 'configurations', 'config.yaml')
    config = read_config(config_path)
    tester = ModelTester(config=config)
    tester.test_net()

import torch
from torch.utils.data import DataLoader
from utils.customdata import CustomDataset
from torch.nn import CrossEntropyLoss
import os

# Loss can be calculated 2 ways
# Loss on single image or on batch
def test_net(path=None):
    try:
        # Creating DataLoader
        test_DataLoader = DataLoader(CustomDataset('test'), batch_size=8, shuffle=False)
        cost = CrossEntropyLoss()
        net = torch.load(path)
        net.eval()

        test_loss = 0
        test_accuracy = 0
        total = 0
        total_batch = 0
        with torch.no_grad():
            for batch, data in enumerate(test_DataLoader):
                images = data[0].to(device)
                labels = data[1].to(device)

                output = net(images)
                loss = cost(output, labels)

                predictions = torch.argmax(output, 1)
                print(f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}",)

                test_loss += loss.item()
                test_accuracy += (predictions == labels).sum().item()
                total_batch +=1
                total += labels.size(0)

        print(f"Model  -->   Loss : {test_loss /total_batch }      Accuracy : {(test_accuracy/total)*100} %")

    except Exception as e:
        raise e


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Inferencing Using Device : {device} \n')
    path = os.path.join('.', 'training_result', 'Model._2022_03_30_12_21', 'model._2022_03_30_12_21.pt')
    test_net(path=path)

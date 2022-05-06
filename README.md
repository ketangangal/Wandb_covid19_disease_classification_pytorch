## 🚀 COVID-19, Viral Pneumonia, Normal using pytorch

Image classification with wandb integration for logging of params and visualization.
With a few lines of code, wandb saves your model's hyperparameters and output metrics and gives you all visual charts like for training, comparison of model, accuracy, etc.
It automatically tracks the state of your code, system metrics and configuration parameters

![image](https://user-images.githubusercontent.com/40850370/167085211-f6805639-3ea7-40fa-a072-3ba76ad990a0.png)

## 💥 Initial Setup 
1. setup env and install dependencies.
```
git clone https://github.com/ketangangal/Wandb_covid19_disease_classification_pytorch.git
conda create --prefix ./env python=3.9
conda activate ./env

pip install -r requirements.txt 
```

## 🌟 Folder Structure 

![image](https://user-images.githubusercontent.com/40850370/167084687-d4ab4deb-769f-41e0-ba70-2ffb9c6bdac0.png)

### 💁 Dataset Link : https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset?resource=download


## 😱  Wandb Setup 
```commandline
Create account
copy Api Key 

pip install wandb
wandb login

enter api key to start tracking.
```

## ✨ Run Project
#### run train.py to train model and results will be saved in training_results folder
```commandline
python src/train.py
```

#### run test.py to get predictions
```commandline
python src/test.py
```

## Wandb Documentation
https://docs.wandb.ai/ref/python/data-types

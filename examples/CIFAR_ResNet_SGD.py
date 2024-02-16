import torch
import torch.nn as nn


import sys
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
from utils import class_accuracy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data.dataloader import default_collate

import pandas as pd

from data_utils import read_data_sets, read_cifar
from train_utils import train_model

torch.set_default_dtype(torch.float32)

sys.path.append("../src")

from optim.FishLeg import FISH_LIKELIHOODS


seed = 13
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = read_cifar("../data/", if_autoencoder=False)

## Dataset
train_dataset = dataset.train
test_dataset = dataset.test

batch_size = 500

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

aux_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)


model = models.resnet18(weights="DEFAULT")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

likelihood = FISH_LIKELIHOODS["softmax"](device=device)
lr = 0.01
weight_decay = 5e-4

log_dir = f"runs/CIFAR_ResNet18_pretrained_SGD/lr={lr}_lambda={weight_decay}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

writer = SummaryWriter(
    log_dir=log_dir,
)

opt = optim.SGD(
    model.parameters(), 
    lr=lr,
    momentum=0.9, 
    weight_decay=weight_decay
)

epoch = 100
trained_model, train_df_per_step, test_df_per_step, df_per_epoch = train_model(model, train_loader, test_loader, opt, likelihood,class_accuracy, epochs=epoch, device=device, savedir=log_dir, writer=writer)
pd.DataFrame.to_csv(test_df_per_step, f"{log_dir}/test_df_per_step.csv")
pd.DataFrame.to_csv(train_df_per_step, f"{log_dir}/train_df_per_step.csv")

print("Finished Training")



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

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS, initialise_FishModel


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

batch_size = 2000

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

aux_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2000, shuffle=False,
    # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)


model = models.resnet18(weights="DEFAULT")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)


likelihood = FISH_LIKELIHOODS["softmax"](device=device)
lr = 0.01
beta = 0.9
weight_decay = 1e-5

aux_lr = 1e-5
aux_eps = 1e-8
scale_factor = 1
damping = 0.1
update_aux_every = 1

method = "antithetic"
precondition_aux = False

model = initialise_FishModel(
    model, module_names="__ALL__", fish_scale=scale_factor / damping
)

model = model.to(device)

log_dir = f"runs/experiment/CIFAR_ResNet18_pretrained_fishleg/lr={lr}_batch_size={batch_size}_lambda={weight_decay}_aux_lr={aux_lr}_damping={damping}_update_aux={update_aux_every}_method={method}_aux_eps={aux_eps}_precond={precondition_aux}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

writer = SummaryWriter(
    log_dir=log_dir,
)

opt = FishLeg(
    model,
    aux_loader,
    likelihood,
    lr=lr,
    beta=beta,
    weight_decay=weight_decay,
    aux_lr=aux_lr,
    aux_betas=(0.9, 0.999),
    aux_eps=aux_eps,
    damping=damping,
    update_aux_every=update_aux_every,
    writer=writer,
    method=method,
    method_kwargs={"eps": 1e-4},
    precondition_aux=precondition_aux,
    device=device,
    aux_log = True
)

epoch = 100
trained_model, train_df_per_step, test_df_per_step, df_per_epoch = train_model(model, train_loader, test_loader, opt, likelihood,class_accuracy, epochs=epoch, device=device, savedir=log_dir, writer=writer)
pd.DataFrame.to_csv(test_df_per_step, f"{log_dir}/test_df_per_step.csv")
pd.DataFrame.to_csv(train_df_per_step, f"{log_dir}/train_df_per_step.csv")

print("Finished Training")



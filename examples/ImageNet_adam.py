import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import time
import os
import sys
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
from utils import class_accuracy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data.dataloader import default_collate



from data_utils import read_data_sets, get_tinyImageNet, dense_to_one_hot

torch.set_default_dtype(torch.float32)

sys.path.append("../src")

from optim.FishLeg import FISH_LIKELIHOODS


seed = 13
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = get_tinyImageNet()

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


# model = models.resnet18(weights='DEFAULT')
# # model.avgpool = nn.AdaptiveAvgPool2d(1)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 200)

# # Freeze all layers
# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze last layer
# for param in model.fc.parameters():
#     param.requires_grad = True

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)


model = model.to(device)

likelihood = FISH_LIKELIHOODS["softmax"](device=device)
lr = 0.001
# betas = (0.7, 0.9)
weight_decay = 1e-5
# eps = 1e-8

writer = SummaryWriter(
    log_dir=f"runs/ImageNet_adam/lr={lr}_lambda={weight_decay}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

opt = optim.Adam(
    model.parameters(),
    lr=lr,
    # betas=betas,
    weight_decay=weight_decay,
    # eps=eps,
)


epochs = 20

st = time.time()
eval_time = 0

for epoch in range(1, epochs + 1):
    with tqdm(train_loader, unit="batch") as tepoch:
        running_loss = 0
        running_acc = 0
        for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):
            tepoch.set_description(f"Epoch {epoch}")
            batch_data, batch_labels = (
                batch_data.to(device),
                batch_labels.to(device),
            )
            batch_labels = dense_to_one_hot(batch_labels, max_value=199, min_value=0, device=device)
            opt.zero_grad()
            output = model(batch_data)

            loss = likelihood(output, batch_labels)

            running_loss += loss.item()

            running_acc += class_accuracy(output, batch_labels).item()

            loss.backward()
            opt.step()

            et = time.time()
            if n % 50 == 0:
                model.eval()

                running_test_loss = 0
                running_test_acc = 0

                for m, (test_batch_data, test_batch_labels) in enumerate(test_loader, start=1):
                    test_batch_data, test_batch_labels = (
                        test_batch_data.to(device), 
                        test_batch_labels.to(device)
                        )
                    
                    test_batch_labels = dense_to_one_hot(test_batch_labels, max_value=199, min_value=0, device=device)

                    test_output = model(test_batch_data)

                    test_loss = likelihood(test_output, test_batch_labels)

                    running_test_loss += test_loss.item()

                    running_test_acc += class_accuracy(
                        test_output, test_batch_labels
                    ).item()

                running_test_loss /= m
                running_test_acc /= m

                tepoch.set_postfix(
                    acc=100 * running_acc / n, test_acc=running_test_acc * 100
                )
                model.train()
                eval_time += time.time() - et

        epoch_time = time.time() - st - eval_time

        tepoch.set_postfix(
            loss=running_loss / n, test_loss=running_test_loss, epoch_time=epoch_time
        )
        # Write out the losses per epoch
        writer.add_scalar("Acc/train", 100 * running_acc / n, epoch)
        writer.add_scalar("Acc/test", 100 * running_test_acc, epoch)

        # Write out the losses per epoch
        writer.add_scalar("Loss/train", running_loss / n, epoch)
        writer.add_scalar("Loss/test", running_test_loss, epoch)

        # Write out the losses per wall clock time
        writer.add_scalar("Loss/train/time", running_loss / n, epoch_time)
        writer.add_scalar("Loss/test/time", running_test_loss, epoch_time)




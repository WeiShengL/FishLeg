import pandas as pd
import time
import os
from tqdm import tqdm
import torch

def train_model(model, train_loader, test_loader, opt, likelihood, class_accuracy, epochs=100, device='cuda', savedir=False, writer=None):
    '''
    Function to train model and obtain metrics per step and per epoch

    Inputs:
        model: model to train
        train_loader: training data loader
        test_loader: test data loader
        opt: optimiser
        likelihood: likelihood function
        epochs: number of epochs to train for
        device: device to train on

    Outputs:
        model: trained model
        train_df_per_step: dataframe of training loss, accuracy and time per step
        test_df_per_step: dataframe of test loss, accuracy and time per step
        df_per_epoch: dataframe of training and test loss, accuracy and time per epoch 
    '''
    train_df_per_step = pd.DataFrame(columns=['loss', 'acc', 'step_time', 'aux_loss'])
    test_df_per_step = pd.DataFrame(columns=['loss', 'acc'])
    df_per_epoch = pd.DataFrame(columns=['train_loss', 'train_acc', 'epoch_time', 'test_loss', 'test_acc'])
    st = time.time()
    eval_time = 0

    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            running_loss = 0
            running_acc = 0
            running_aux_loss = 0
            for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):
                tepoch.set_description(f"Epoch {epoch}")

                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                opt.zero_grad()
                output = model(batch_data)
                loss = likelihood(output, batch_labels)

                running_loss += loss.item()
                running_acc += class_accuracy(output, batch_labels).item()

                loss.backward()
                opt.step()

                et = time.time()     
                try:
                    aux_loss = opt.aux_loss
                    df_temp = pd.DataFrame([[loss.item(), class_accuracy(output, batch_labels).item(), et-st, aux_loss.item()]], columns=['loss', 'acc', 'step_time', 'aux_loss'])
                    if not pd.isna(aux_loss):  # Check if aux_loss is not NaN
                        running_aux_loss += opt.aux_loss.item()

                except:
                    df_temp = pd.DataFrame([[loss.item(), class_accuracy(output, batch_labels).item(), et-st]], columns=['loss', 'acc', 'step_time'])

                if train_df_per_step.empty:
                    train_df_per_step = df_temp
                else:
                    train_df_per_step = pd.concat([train_df_per_step, df_temp], ignore_index=True)
                if n % 50 == 0:
                    model.eval()

                    running_test_loss = 0
                    running_test_acc = 0

                    for m, (test_batch_data, test_batch_labels) in enumerate(test_loader, start=1):
                        test_batch_data, test_batch_labels = test_batch_data.to(device), test_batch_labels.to(device)

                        test_output = model(test_batch_data)

                        test_loss = likelihood(test_output, test_batch_labels)

                        running_test_loss += test_loss.item()
                        running_test_acc += class_accuracy(test_output, test_batch_labels).item()

                        df_temp = pd.DataFrame([[test_loss.item(), class_accuracy(test_output, test_batch_labels).item()]], columns=['loss', 'acc'])
                        if test_df_per_step.empty:
                            test_df_per_step = df_temp
                        else:
                            test_df_per_step = pd.concat([test_df_per_step, df_temp], ignore_index=True)

                    running_test_loss /= m
                    running_test_acc /= m

                    tepoch.set_postfix(acc=100 * running_acc / n, test_acc=running_test_acc * 100)
                    model.train()
                    eval_time += time.time() - et
            
            epoch_time = time.time() - st - eval_time
            tepoch.set_postfix(loss=running_loss / n, test_loss=running_test_loss, epoch_time=epoch_time)


            df_temp = pd.DataFrame([[running_loss / n, 100 * running_acc / n, epoch_time, running_test_loss, 100 * running_test_acc, running_aux_loss/n]], columns=['train_loss', 'train_acc', 'epoch_time', 'test_loss', 'test_acc', 'aux_loss'])

            if df_per_epoch.empty:
                df_per_epoch = df_temp
            else:
                df_per_epoch = pd.concat([df_per_epoch, df_temp], ignore_index=True)

            if savedir:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                if not os.path.exists(f"{savedir}/ckpts"):
                    os.makedirs(f"{savedir}/ckpts")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': opt.state_dict(),
                    'metrics': df_per_epoch,
                    }, f"{savedir}/ckpts/epoch={epoch}-test_loss={round(running_test_loss, 4)}.pt")
                

            if writer:
                # Write out the losses per epoch
                writer.add_scalar("Acc/train", 100 * running_acc / n, epoch)
                writer.add_scalar("Acc/test", 100 * running_test_acc, epoch)

                # Write out the losses per epoch
                writer.add_scalar("Loss/train", running_loss / n, epoch)
                writer.add_scalar("Loss/test", running_test_loss, epoch)

                # Write out the losses per wall clock time
                writer.add_scalar("Loss/train/time", running_loss / n, epoch_time)
                writer.add_scalar("Loss/test/time", running_test_loss, epoch_time)

    return model, train_df_per_step, test_df_per_step, df_per_epoch
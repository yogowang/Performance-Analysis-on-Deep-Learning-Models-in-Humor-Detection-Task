import numpy as np
from torch import device
import torch.optim
from tqdm import tqdm
import torch
import models

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def train(train_iter, dev_iter, model, number_epoch ,loss_fn,optimizer):
    for epoch in range(1, number_epoch + 1):
        model.train()
        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far
        for batch in tqdm(train_iter):
            feature, target = batch

            # print(feature)
            feature, target = feature.to(device), target.to(device)

            no_observations = no_observations + target.shape[0]

            predictions = model(feature).squeeze(1)
            optimizer.zero_grad()
            loss = loss_fn(predictions, target)

            loss.backward()
            optimsse, __ = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())
            optimizer.step()

            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += optimsse
        print()
        print('model training is completed')
        print()
        valid_loss, valid_mse, __, __ = eval(dev_iter, model,loss_fn)

        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        print(
            f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse ** 0.5:.2f} | \
                    Val. Loss: {valid_loss:.2f} | Val. MSE: {valid_mse:.2f} |  Val. RMSE: {valid_mse ** 0.5:.2f} |')
    model.eval()
def train1(train_iter, dev_iter, model, number_epoch,loss_fn,optimizer):
    for epoch in range(1, number_epoch + 1):
        model.train()
        bptt=32
        src_mask=models.generate_square_subsequent_mask(bptt).to(device)
        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far
        for batch in tqdm(train_iter):
            feature, target = batch
            if bptt>len(feature):
                bptt=len(feature)
                src_mask = models.generate_square_subsequent_mask(bptt).to(device)
            feature, target = feature.to(device), target.to(device)
            no_observations = no_observations + target.shape[0]
            predictions = model(feature,src_mask).squeeze(1)
            optimizer.zero_grad()
            loss = loss_fn(predictions, target)
            loss.backward()
            optimsse, __ = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())
            optimizer.step()

            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += optimsse

        print()
        print('model training is completed')
        print()
        valid_loss, valid_mse, __, __ = eval1(dev_iter, model,loss_fn)
        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        print(
            f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse ** 0.5:.2f} | \
                    Val. Loss: {valid_loss:.2f} | Val. MSE: {valid_mse:.2f} |  Val. RMSE: {valid_mse ** 0.5:.2f} |')
    model.eval()
def eval1(dev_iter,model,loss_fn):
    """
    Evaluating model performance on the dev set
    """
    # TO DO
    model.eval()
    epoch_loss = 0
    epoch_sse = 0
    pred_all = []
    trg_all = []
    no_observations = 0
    bptt=32
    src_mask = models.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for batch in dev_iter:
            feature, target = batch
            if bptt>len(feature):
                bptt=len(feature)
                src_mask = models.generate_square_subsequent_mask(bptt).to(device)
            feature, target = feature.to(device), target.to(device)

            no_observations = no_observations + target.shape[0]

            predictions = model(feature,src_mask).squeeze(1)
            loss = loss_fn(predictions, target)

            # We get the mse
            pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
            sse, __ = model_performance(pred, trg)

            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += sse
            pred_all.extend(pred)
            trg_all.extend(trg)

    return epoch_loss / no_observations, epoch_sse / no_observations, np.array(pred_all), np.array(trg_all)

def eval(dev_iter, model,loss_fn):
    """
    Evaluating model performance on the dev set
    """
    # TO DO
    model.eval()
    epoch_loss = 0
    epoch_sse = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for batch in dev_iter:
            feature, target = batch

            feature, target = feature.to(device), target.to(device)

            no_observations = no_observations + target.shape[0]

            predictions = model(feature).squeeze(1)
            loss = loss_fn(predictions, target)

            # We get the mse
            pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
            sse, __ = model_performance(pred, trg)

            epoch_loss += loss.item() * target.shape[0]
            epoch_sse += sse
            pred_all.extend(pred)
            trg_all.extend(trg)

    return epoch_loss / no_observations, epoch_sse / no_observations, np.array(pred_all), np.array(trg_all)

def test1(test_iter, model,loss_fn):
    """
    Evaluating model performance on the dev set
    """
    print(1)
    # TO DO
    model.eval()
    no_observations = 0
    bptt = 32
    src_mask = models.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for batch in test_iter:
            feature, target = batch
            if bptt>len(feature):
                bptt=len(feature)
                src_mask = models.generate_square_subsequent_mask(bptt).to(device)
            feature, target = feature.to(device), target.to(device)

            no_observations = no_observations + target.shape[0]

            predictions = model(feature,src_mask).squeeze(1)
            loss = loss_fn(predictions, target)
            acc = accuracy(predictions, target)
            print(f'Test Loss: {loss:.3f} | Test Acc: {acc * 100:.2f}%')
def test(test_iter, model,loss_fn):
    """
    Evaluating model performance on the dev set
    """
    print(1)
    # TO DO
    model.eval()
    no_observations = 0

    with torch.no_grad():
        for batch in test_iter:
            feature, target = batch

            feature, target = feature.to(device), target.to(device)

            no_observations = no_observations + target.shape[0]

            predictions = model(feature).squeeze(1)
            loss = loss_fn(predictions, target)
            acc = accuracy(predictions, target)
            print(f'Test Loss: {loss:.3f} | Test Acc: {acc * 100:.2f}%')

def model_performance(output, target, print_output=False):
    """
    Returns SSE and MSE per batch (printing the MSE and the RMSE)
    """

    sq_error = (output - target) ** 2

    sse = np.sum(sq_error)
    mse = np.mean(sq_error)
    rmse = np.sqrt(mse)

    if print_output:
        print(f'| MSE: {mse:.2f} | RMSE: {rmse:.2f} |')

    return sse, mse

def accuracy(output, target):
    # print('output:', output, 'target: ', target)
    #####################################
    # Q: Return the accuracy of the model
    #####################################
    # Pass through the sigmoid and round the values to 0 or 1
    output = torch.round(torch.sigmoid(output))
    correct = (output == target).float()
    acc = correct.mean()

    return acc
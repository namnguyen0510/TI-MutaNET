import torch
from sklearn.metrics import *

def train(model, x_train, y_train, criterion, optimizer, schedule, report = False):
    correct = 0.
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    schedule.step()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    correct = (output == y_train).float().sum()
    if report:
        print("TRAIN LOSS: {}".format(loss.item()))
        print("TRAIN ACC: {}".format(100*correct/len(x_train)))
        print("CLASSIFICATION REPORT")
        print(classification_report(y_train.detach().cpu().numpy(),output.detach().cpu().numpy(), digits = 4))
    return loss.item(), correct/len(x_train)

def test(model, x_test, y_test, criterion, report = False):
    correct = 0.
    model.eval()
    output = model(x_test)
    loss = criterion(output, y_test)
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    correct = (output == y_test).float().sum()
    if report:
        print("TEST LOSS: {}".format(loss.item()))
        print("TEST ACC: {}".format(100*correct/len(x_test)))
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test.detach().cpu().numpy(),output.detach().cpu().numpy(), digits = 4))
    return loss.item(), correct/len(x_test)

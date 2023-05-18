import numpy as np
import csv
import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from SeafoodMlpKit import DataLoader_BackHead, computeAndsave, MLP, balance_dataset



x_all, y_all = DataLoader_BackHead("../Dataset_badminton/transfer_t/", "../Dataset_badminton/train/")
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all.reshape(-1, 1), test_size=0.2, random_state=42)

x_test = torch.from_numpy(x_test).float().cuda()
y_test = torch.from_numpy(y_test).float().cuda()

print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

#Train mlp model
import torch
import torch.nn as nn
import torch.optim as optim

# Define the hyperparameters and load the data
input_dim = 36
output_dim = 1
learning_rate = 0.001
num_epochs = 100000
batch_size = 64

# Create the MLP and specify the loss function and optimizer
mlp = MLP(input_dim, output_dim).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)#optim.Adam(mlp.parameters(), lr=learning_rate)

# Train the MLP
for epoch in trange(num_epochs):
    if epoch%100==0:
        x_train, y_train = balance_dataset(x_all, y_all)
        y_train = y_train.reshape(-1, 1)
        

        # Move the data to the GPU device
        x_train = torch.from_numpy(x_train).float().cuda()
        y_train = torch.from_numpy(y_train).float().cuda()

        # Convert the data to PyTorch tensors
        x_tensor = x_train
        y_tensor = y_train
        
    running_loss = 0.0
    for i in range(0, x_train.shape[0], batch_size):
        # Get the current batch of data
        x_batch = x_tensor[i:i+batch_size, :]
        y_batch = y_tensor[i:i+batch_size, :]

        # Zero the gradients and compute the forward pass
        optimizer.zero_grad()
        y_pred = mlp(x_batch)

        # Compute the loss and perform backpropagation
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = running_loss / (x_train.shape[0] / batch_size)
    

    if epoch%100==0:
        model_path = "./Backheadckp/pose_mlp_"+str(epoch)+".pth"
        acc = computeAndsave(mlp, x_test, y_test, model_path)
        print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, num_epochs, avg_loss))
        print(acc)

model_path = "./Backheadckp/pose_mlp_lastone.pth"
acc = computeAndsave(mlp, x_test, y_test, model_path)
print("Accuracy:", acc)
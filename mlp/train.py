#LoadDataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import minmax_scale
import SeafoodMlpKit as smk
from tqdm import tqdm, trange



root = "../Dataset_badminton/mp_5poser"
dataset = smk.DataLoader(root)
dataset = smk.balance_classes(dataset)
print("loading dataset..")
x_train=[]
progress = tqdm(total=len(dataset[:, 0]))
for p in dataset[:, 0]:
    progress.update(1)
    arr = np.loadtxt(p, delimiter=',')
    sqrt_arr = arr.reshape(1, 85, 3)
    #print(sqrt_arr.shape)
    #flat_arr = arr.flatten()
    x_train.append(sqrt_arr.tolist())
    #print(p)
x_all = np.array(x_train)
y_all = dataset[:, 1].astype(int).reshape(-1, 1)

#MinMax Scaler
#x_all = minmax_scale(x_all)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)


#Train mlp model
import torch
import torch.nn as nn
import torch.optim as optim

# Define the hyperparameters and load the data
input_dim = 5*17*3
hidden_dim = 1024
output_dim = 1
learning_rate = 0.01
num_epochs = 100000
batch_size = 2000

# Move the data to the GPU device
x_train = torch.from_numpy(x_train).float().cuda()
y_train = torch.from_numpy(y_train).float().cuda()
x_test = torch.from_numpy(x_test).float().cuda()
y_test = torch.from_numpy(y_test).float().cuda()

# Convert the data to PyTorch tensors
x_tensor = x_train
y_tensor = y_train

# Create the MLP and specify the loss function and optimizer
#mlp = smk.MLP(input_dim, hidden_dim, output_dim).cuda()
mlp = smk.MyCNN().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)#optim.Adam(mlp.parameters(), lr=learning_rate)

# Train the MLP
for epoch in trange(num_epochs):
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
        model_path = "./checkpoints/pose_mlp_"+str(epoch)+".pth"
        acc = smk.computeAndsave(mlp, x_test, y_test, model_path)
        print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, num_epochs, avg_loss))
        print(acc)

model_path = "./checkpoints/pose_mlp_lastone.pth"
acc = smk.computeAndsave(mlp, x_test, y_test, model_path)
print("Accuracy:", acc)
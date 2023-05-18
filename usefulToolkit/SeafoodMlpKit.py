import numpy as np
import csv
import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 2048)
        self.hidden2 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1))
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=85*3*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        
        # Define the activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        
        # Pass input through convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 85*3*32)
        
        # Pass input through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def computeAndsave(mlp, x_test, y_test, model_path):
    # Compute the predicted labels for the test data
    with torch.no_grad():
        y_pred = mlp(x_test)
        y_pred = (torch.sigmoid(y_pred) >= 0.5).float()

    # Compute the accuracy
    acc = (y_pred == y_test).float().mean()

    s = 'Accuracy: {:.4f}'.format(acc)


    #model_path = "./pose_mlp.pt"
    torch.save(mlp, model_path)
    print(model_path, "saved!")

    return s


def csv2np(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    data_array = np.array(data)
    return data_array


def DataLoader(root):
    rootFolders = os.listdir(root)

    #x_train = [frame, whosHit, np.array(pose)]
    #y_train = [RoundHead, Backhand, BallHeight, LandingX, LandingY, HitterLocationX, HitterLocationY, DefenderLocationX, DefenderLocationY, BallType, Winner]

    #[[x_train], isHit?]
    #[[x_train], [y_train]]

    dataset=[]
    progress = tqdm(total=len(rootFolders))

    for folders in rootFolders:
        progress.update(1)
        data = os.path.join(root, folders)
        dataFolders = os.listdir(data)

        #Load y_train
        dataFolders.remove("classes.csv")
        csvPath = os.path.join(data, "classes.csv")
        y_train = csv2np(csvPath)[1:, 1:]

        #Load x_train
        x_train=[]
        for dataname in dataFolders:
            if dataname.endswith(".txt"):
                name = dataname.replace(".txt", "")
                f, fNum, p = name.split("_")
                if p=='1': P='A'
                elif p=='0': P='B'
                txtPath = os.path.join(data, dataname)

                x_train.append([fNum, P, txtPath])
        x_train = np.array(x_train)

        #print(y_train)
        #print(x_train)
        #print(y_train[:, 0])
        #print(x_train[:, 0])
        

        
        #for y in y_train:
        for x in x_train:
            y_list = y_train[:, 0].tolist()
            #print(y_list)
            if (x[0] in y_list)or( str(int(x[0])-1) in y_list )or( str(int(x[0])+1) in y_list ):
                index = y_list.index(x[0])
                if (x[1]==y_train[index][1]):
                    r = [x[2], 1]
                else:
                    r = [x[2], 0]
            else:
                r = [x[2], 0]
            dataset.append(r)
            #print(r)
        #print(a)
        #print(folders, "loaded")
    dataset = np.array(dataset)

    
    return dataset

def balance_classes(dataset):
    # Find the counts of class 0 and class 1 in the dataset
    class_0_count = np.count_nonzero(dataset[:, 1] == "0")
    print("class_0_count:", class_0_count)
    class_1_count = np.count_nonzero(dataset[:, 1] == "1")
    print("class_1_count:", class_1_count)
    
    # If class 0 and class 1 are already balanced, return the dataset
    if class_0_count == class_1_count:
        return dataset
    
    # Find the indices of the majority class
    majority_class = "0" if class_0_count > class_1_count else "1"
    majority_class_indices = np.where(dataset[:, 1] == majority_class)[0]
    
    # Randomly select samples to remove from the majority class
    samples_to_remove = np.random.choice(majority_class_indices, size=abs(class_0_count-class_1_count), replace=False)
    
    # Delete the selected samples and return the balanced dataset
    balanced_dataset = np.delete(dataset, samples_to_remove, axis=0)
    class_0_count = np.count_nonzero(balanced_dataset[:, 1] == "0")
    print("balanced_dataset_0_count:", class_0_count)
    class_1_count = np.count_nonzero(balanced_dataset[:, 1] == "1")
    print("balanced_dataset_1_count:", class_1_count)
    return balanced_dataset

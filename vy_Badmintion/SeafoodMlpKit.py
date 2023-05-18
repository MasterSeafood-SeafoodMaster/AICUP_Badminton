import numpy as np
import csv
import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x

import torch.nn as nn

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


def DataLoader(txtroot, csvroot):
    #txtroot = "../Dataset_badminton/transfer/"
    #csvroot = "../Dataset_badminton/train/"

    X=[]; y=[]
    rootFolders = os.listdir(txtroot)
    progress = tqdm(total=len(rootFolders))
    for folders in rootFolders:
        progress.update(1)
        txtdata = os.path.join(txtroot, folders)
        dataFolders = os.listdir(txtdata)

        for dataname in dataFolders:
            dpath = os.path.join(txtdata, dataname)
            if dataname==folders+"_court.txt":
                court_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                court_arr = norm(court_arr)

            elif dataname==folders+"_p0.txt":
                p0_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                p0_arr = norm(p0_arr)

            elif dataname==folders+"_p1.txt":
                p1_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                p1_arr = norm(p1_arr)

        csvdata = os.path.join(csvroot, folders)
        dataFolders = os.listdir(csvdata)
        for dataname in dataFolders:
            dpath = os.path.join(csvdata, dataname)
            if dataname == folders+"_S2.csv":
                classes = csv2np(dpath)

        #[net, court, ball]
        net = court_arr[:, 0:4]
        court = court_arr[:, 4:12]
        ball = court_arr[:, 12:14]

        low = min(court_arr.shape[0], p0_arr.shape[0], p1_arr.shape[0])
        HitFrame = classes[1:, 1].astype(int)
        WhosHit = classes[1:, 2].astype(str)
        noHitFrame = get_other_values(HitFrame, 0, low-1)

        
        for i, f in enumerate(HitFrame):
            if f<low:
                ball10 = get_ball_coordinates(ball, f).reshape(22, )
                if WhosHit[i]=="A":
                    X.append(p0_arr[f].tolist()+ball10.tolist())
                    y.append(1)
                elif WhosHit[i]=="B":
                    X.append(p1_arr[f].tolist()+ball10.tolist())
                    y.append(1)

        for i, f in enumerate(noHitFrame):
            if f<low:
                ball10 = get_ball_coordinates(ball, f).reshape(22, )
                r = random.randint(0, 2)
                if r==0:
                    X.append(p0_arr[f].tolist()+ball10.tolist())
                    y.append(0)
                elif r==1:
                    X.append(p1_arr[f].tolist()+ball10.tolist())
                    y.append(0)

    X = np.array(X)
    y = np.array(y)
    #X, y = balance_dataset(X, y)
    print(X.shape)
    print(y.shape)
    print(calculate_class_distribution(y))
    return X, y

def DataLoader_RoundHead(txtroot, csvroot):
    #txtroot = "../Dataset_badminton/transfer/"
    #csvroot = "../Dataset_badminton/train/"

    X=[]; y=[]
    rootFolders = os.listdir(txtroot)
    progress = tqdm(total=len(rootFolders))
    for folders in rootFolders:
        progress.update(1)
        txtdata = os.path.join(txtroot, folders)
        dataFolders = os.listdir(txtdata)

        for dataname in dataFolders:
            dpath = os.path.join(txtdata, dataname)
            if dataname==folders+"_court.txt":
                court_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                court_arr = norm(court_arr)

            elif dataname==folders+"_p0.txt":
                p0_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                p0_arr = norm(p0_arr)

            elif dataname==folders+"_p1.txt":
                p1_arr = np.loadtxt(dpath, delimiter=',').astype(float)
                p1_arr = norm(p1_arr)

        csvdata = os.path.join(csvroot, folders)
        dataFolders = os.listdir(csvdata)
        for dataname in dataFolders:
            dpath = os.path.join(csvdata, dataname)
            if dataname == folders+"_S2.csv":
                classes = csv2np(dpath)

        #[net, court, ball]
        net = court_arr[:, 0:4]
        court = court_arr[:, 4:12]
        ball = court_arr[:, 12:14]

        low = min(court_arr.shape[0], p0_arr.shape[0], p1_arr.shape[0])
        HitFrame = classes[1:, 1].astype(int)
        WhosHit = classes[1:, 2].astype(str)
        RoundHead = classes[1:, 3].astype(str)

        for i, f in enumerate(HitFrame):
            if WhosHit[i]=="A":
                X.append(p0_arr[f].tolist()+ball[f].tolist())
            elif WhosHit[i]=="B":
                X.append(p1_arr[f].tolist()+ball[f].tolist())

            if RoundHead=="1":
                y.append(0)
            elif RoundHead=="2":
                y.append(1)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    #X, y = balance_dataset(X, y)
    return X, y


from collections import Counter
def balance_dataset(x, y):
    counter = Counter(y)  # 計算每個類別的數量
    min_count = min(counter.values())  # 最小類別數量

    # 建立一個新的平衡後的資料集
    balanced_x = []
    balanced_y = []
    for class_label, count in counter.items():
        indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(indices, size=min_count, replace=False)
        balanced_x.append(x[selected_indices])
        balanced_y.append(y[selected_indices])

    balanced_x = np.concatenate(balanced_x, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    # 將資料集打亂順序
    random_indices = np.random.permutation(len(balanced_x))
    balanced_x = balanced_x[random_indices]
    balanced_y = balanced_y[random_indices]

    return balanced_x, balanced_y


def generate_random_array(low, high, arr):
    size = arr.size*2
    unique_values = set(arr.flatten())  # 將輸入陣列扁平化並轉換為集合，以檢查重複值

    if len(unique_values) > size:
        size = len(unique_values)-1


    result = np.random.choice(range(low, high+1), size, replace=False)
    retry=0
    while any(np.isin(result, arr)):
        result = np.random.choice(range(low, high+1), size, replace=False)
        if retry>1000:
            return []
            break
        else:
            retry+=1

    #sorted_result = np.random.permutation(result)  # 重新排序亂數陣列

    return result


def expand_array(arr, Min, Max):
    expanded_arr = []
    for element in arr:
        expanded_arr += list(np.arange(element-2, element+3))
    expanded_arr = np.array(expanded_arr)
    expanded_arr = np.clip(expanded_arr, Min, Max).astype(int)
    
    return expanded_arr

def norm(arr):
    result = arr.copy().astype(float)  # 創建輸入陣列的副本，以避免修改原始陣列

    result[::2] = result[::2] / 1280  # 偶數項除以1280
    result[1::2] = result[1::2] / 720  # 基數項除以720

    return result

def calculate_class_distribution(arr):
    unique_classes, counts = np.unique(arr, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    return class_distribution


def get_other_values(arr, lower_bound, upper_bound):
    # 找出不在指定範圍內的數字
    other_values = np.setdiff1d(np.arange(lower_bound, upper_bound+1), arr)
    
    return other_values

def get_ball_coordinates(arr, f):
    start_index = max(f - 5, 0)
    end_index = min(f + 5, len(arr) - 1)
    ball = arr[start_index:end_index+1].astype(float)

    if len(ball) < 11:
        padding_length = 11 - len(ball)
        if start_index == 0:
            padding = np.full((padding_length, 2), arr[0])
            ball = np.concatenate((padding, ball))
        else:
            padding = np.full((padding_length, 2), arr[-1])
            ball = np.concatenate((ball, padding))

    return ball

def get_ball_coordinates_f10(arr, f):
    start_index = max(f - 0, 0)
    end_index = min(f + 10, len(arr) - 1)
    ball = arr[start_index:end_index+1]

    if len(ball) < 11:
        padding_length = 11 - len(ball)
        if start_index == 0:
            padding = np.full((padding_length, 2), arr[0])
            ball = np.concatenate((padding, ball))
        else:
            padding = np.full((padding_length, 2), arr[-1])
            ball = np.concatenate((ball, padding))

    return ball
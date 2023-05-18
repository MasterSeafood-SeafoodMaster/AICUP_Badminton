import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

mlp = torch.load("./checkpoints/pose_mlp_200.pth")
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size, encode_size, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, encode_size)
        self.bn1 = nn.BatchNorm1d(input_size)

    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(self.bn1(x)))
        return self.fc2(h), self.fc3(h)

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class AE(nn.Module):
    def __init__(self, input_size, encode_size, h_dim=128, z_dim=20):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim//2, z_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_dim*2, z_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(z_dim, z_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(z_dim*2, h_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim//2, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, encode_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
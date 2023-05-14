# **
# * ����
# *
# * @author 雁楚
# * @edit 雁楚

import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from .AE_base import VAE, AE

import warnings
warnings.simplefilter('ignore')

class ReconBase():
    def __init__(self, recondataset, device='cpu', spark_context=None):
        self._dataset = recondataset
        self._spark_context = spark_context
        self.output_data = None
        self.model = None
        self.dataloader = None
        print("Data shape: {}".format(recondataset.data.shape))

        self.input_size = len(recondataset.covariates) + len(recondataset.treat) + len(recondataset.causal_name)
        self.encode_size = len(recondataset.covariates) + len(recondataset.treat)
        self.device=device

    def get_Dataloader(self, batch_size=256, shuffle=False):
        data_input = torch.tensor(np.array(self._dataset.get_origin_x())).to(torch.float32)
        data_loader = DataLoader(dataset=data_input,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
        self.dataloader = data_loader

    def Recon_fit(self, reconstructor='VAE', z_dim=20, num_epochs=1, learning_rate=1e-2):
        if reconstructor == 'VAE':
            self.model = VAE(input_size=self.input_size, encode_size=self.encode_size).to(self.device)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
            # 创建优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                for i, x in enumerate(self.dataloader):
                    # 获取样本，并前向传播
                    x = x.to(self.device).view(-1, self.input_size)
                    #         print(x.shape, type(x))
                    x_reconst, mu, log_var = self.model(x)

                    # batch_norm = nn.BatchNorm1d(self.encode_size)
                    # x_true = batch_norm(x[:, :self.encode_size])
                    x_true = x[:, :self.encode_size]

                    # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）
                    # KL散度的计算可以参考论文或者文章开头的链接
                    reconst_loss = F.mse_loss(x_reconst, x_true, size_average=False)
                    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                    # 反向传播和优化
                    loss = reconst_loss + kl_div
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % 10 == 0:
                        print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                               .format(epoch+1, num_epochs, i+1, len(self.dataloader), reconst_loss.item(), kl_div.item()))

        elif reconstructor == 'AE':
            self.model = AE(input_size=self.input_size, encode_size=self.encode_size).to(self.device)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
            # 创建优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                for i, x in enumerate(self.dataloader):
                    # 获取样本，并前向传播
                    x = x.to(self.device).view(-1, self.input_size)
                    #         print(x.shape, type(x))
                    x_reconst = self.model(x)
                    x_true = x[:, :self.encode_size]

                    reconst_loss = F.mse_loss(x_reconst, x_true, size_average=False)

                    # 反向传播和优化
                    loss = reconst_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % 10 == 0:
                        print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
                               .format(epoch+1, num_epochs, i+1, len(self.dataloader), reconst_loss.item()))

    def Recon_transform(self, reconstructor = 'VAE'):
        columns = list(list(self._dataset.covariates) + list(self._dataset.treat))
        self.output_data = pd.DataFrame(columns=columns)

        if reconstructor == 'VAE':
            with torch.no_grad():
                for i, x in enumerate(self.dataloader):
                    x = x.to(self.device).view(-1, self.input_size)
                    x_reconst, mu, log_var = self.model(x)
                    self.output_data = self.output_data.append(pd.DataFrame(np.array(x_reconst), columns=columns), ignore_index=True)
        elif reconstructor == 'AE':
            with torch.no_grad():
                for i, x in enumerate(self.dataloader):
                    x = x.to(self.device).view(-1, self.input_size)
                    x_reconst = self.model(x)
                    self.output_data = self.output_data.append(pd.DataFrame(np.array(x_reconst), columns=columns), ignore_index=True)

    def get_reconstruct_data(self):
        return self.output_data

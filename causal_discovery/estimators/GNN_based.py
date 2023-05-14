# **
# * ����
# *
# * @author 雁楚
# * @edit 雁楚

import math
import torch
import argparse
import datetime
import math
import os
import pickle
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG
from .utils import *
from .base import base_discover



_EPS = 1e-10


class MLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(
            self,
            n_in,
            n_xdims,
            n_hid,
            n_out,
            adj_A,
            batch_size,
            do_prob=0.0,
            factor=True,
            tol=0.1,
    ):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True)
        )
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(
            torch.ones_like(torch.from_numpy(adj_A)).double()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        adj_A1 = torch.sinh(3.0 * self.adj_A)

        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()

        H1 = F.relu((self.fc1(inputs)))
        x = self.fc2(H1)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDEncoder(nn.Module):
    def __init__(
            self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1
    ):
        super(MLPDEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True)
        )
        self.factor = factor

        self.Wa = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.fc1 = nn.Linear(n_hid, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)

        n_var = adj_A.shape[0]
        self.embed = nn.Embedding(n_out, n_hid)
        self.dropout_prob = do_prob
        self.alpha = nn.Parameter(
            Variable(torch.div(torch.ones(n_var, n_out), n_out)).double(),
            requires_grad=True,
        )
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(
            torch.ones_like(torch.from_numpy(adj_A)).double()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        adj_A1 = torch.sinh(3.0 * self.adj_A)

        adj_Aforz = preprocess_adj_new(adj_A1)
        adj_A = torch.eye(adj_A1.size()[0]).double()

        bninput = self.embed(inputs.long().view(-1, inputs.size(2)))
        bninput = bninput.view(*inputs.size(), -1).squeeze()
        H1 = F.relu((self.fc1(bninput)))
        x = self.fc2(H1)

        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        prob = my_softmax(logits, -1)
        alpha = my_softmax(self.alpha, -1)

        return (
            x,
            prob,
            adj_A1,
            adj_A,
            self.z,
            self.z_positive,
            self.adj_A,
            self.Wa,
            alpha,
        )


class SEMEncoder(nn.Module):
    """SEM encoder module."""

    def __init__(
            self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0.0, factor=True, tol=0.1
    ):
        super(SEMEncoder, self).__init__()

        self.factor = factor
        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True)
        )
        self.dropout_prob = do_prob
        self.batch_size = batch_size

    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        adj_A1 = torch.sinh(3.0 * self.adj_A)

        adj_A = preprocess_adj_new((adj_A1))
        adj_A_inv = preprocess_adj_new1((adj_A1))

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs - meanF)

        return (
            inputs - meanF,
            logits,
            adj_A1,
            adj_A,
            self.z,
            self.z_positive,
            self.adj_A,
        )


class MLPDDecoder(nn.Module):
    """MLP decoder module. OLD DON"T USE"""

    def __init__(
            self,
            n_in_node,
            n_in_z,
            n_out,
            encoder,
            data_variable_size,
            batch_size,
            n_hid,
            do_prob=0.0,
    ):
        super(MLPDDecoder, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias=True)
        self.out_fc3 = nn.Linear(n_hid, n_out, bias=True)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
            self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        adj_A_new = torch.eye(origin_A.size()[0]).double()
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        adj_As = adj_A_new

        H3 = F.relu(self.out_fc1((mat_z)))

        out = self.out_fc3(H3)

        return mat_z, out, adj_A_tilt


class MLPDiscreteDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(
            self,
            n_in_node,
            n_in_z,
            n_out,
            encoder,
            data_variable_size,
            batch_size,
            n_hid,
            do_prob=0.0,
    ):
        super(MLPDiscreteDecoder, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias=True)

        self.out_fc3 = nn.Linear(n_hid, n_out, bias=True)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.softmax = nn.Softmax(dim=2)

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
            self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        adj_A_new = torch.eye(origin_A.size()[0]).double()
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        adj_As = adj_A_new

        H3 = F.relu(self.out_fc1((mat_z)))

        out = self.softmax(self.out_fc3(H3))

        return mat_z, out, adj_A_tilt


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(
            self,
            n_in_node,
            n_in_z,
            n_out,
            encoder,
            data_variable_size,
            batch_size,
            n_hid,
            do_prob=0.0,
    ):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
            self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt


class SEMDecoder(nn.Module):
    """SEM decoder module."""

    def __init__(
            self,
            n_in_node,
            n_in_z,
            n_out,
            encoder,
            data_variable_size,
            batch_size,
            n_hid,
            do_prob=0.0,
    ):
        super(SEMDecoder, self).__init__()

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print("Using learned interaction net decoder.")

        self.dropout_prob = do_prob

    def forward(
            self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)
        out = mat_z

        return mat_z, out - Wa, adj_A_tilt



class GNN_struct(base_discover):
    def __init__(self, dataset_discover, CONFIG):
        super(GNN_struct, self).__init__(dataset_discover)
        self.CONFIG = CONFIG
        self.CONFIG.cuda = not self.CONFIG.no_cuda and torch.cuda.is_available()
        self.CONFIG.factor = not self.CONFIG.no_factor
        torch.manual_seed(self.CONFIG.seed)
        if self.CONFIG.cuda:
            torch.cuda.manual_seed(self.CONFIG.seed)

    def _get_feature_columns(self):
        return self._dataset.get_data_columns()


    def _get_adjacent_matrix(self):
        train_loader, valid_loader, test_loader, self.CONFIG = load_data(self._dataset, self.CONFIG, self.CONFIG.batch_size)
        off_diag = np.ones([self.CONFIG.data_variable_size, self.CONFIG.data_variable_size]) - np.eye(self.CONFIG.data_variable_size)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.DoubleTensor(rel_rec)
        rel_send = torch.DoubleTensor(rel_send)

        # add adjacency matrix A
        num_nodes = self.CONFIG.data_variable_size
        adj_A = np.zeros((num_nodes, num_nodes))

        if self.CONFIG.encoder == "mlp":
            encoder = MLPEncoder(
                self.CONFIG.data_variable_size * self.CONFIG.x_dims,
                self.CONFIG.x_dims,
                self.CONFIG.encoder_hidden,
                int(self.CONFIG.z_dims),
                adj_A,
                batch_size=self.CONFIG.batch_size,
                do_prob=self.CONFIG.encoder_dropout,
                factor=self.CONFIG.factor,
                ).double()
        elif self.CONFIG.encoder == "sem":
            encoder = SEMEncoder(
                self.CONFIG.data_variable_size * self.CONFIG.x_dims,
                self.CONFIG.encoder_hidden,
                int(self.CONFIG.z_dims),
                adj_A,
                batch_size=self.CONFIG.batch_size,
                do_prob=self.CONFIG.encoder_dropout,
                factor=self.CONFIG.factor,
                ).double()

        if self.CONFIG.decoder == "mlp":
            decoder = MLPDecoder(
                self.CONFIG.data_variable_size * self.CONFIG.x_dims,
                self.CONFIG.z_dims,
                self.CONFIG.x_dims,
                encoder,
                data_variable_size=self.CONFIG.data_variable_size,
                batch_size=self.CONFIG.batch_size,
                n_hid=self.CONFIG.decoder_hidden,
                do_prob=self.CONFIG.decoder_dropout,
                ).double()
        elif self.CONFIG.decoder == "sem":
            decoder = SEMDecoder(
                self.CONFIG.data_variable_size * self.CONFIG.x_dims,
                self.CONFIG.z_dims,
                2,
                encoder,
                data_variable_size=self.CONFIG.data_variable_size,
                batch_size=self.CONFIG.batch_size,
                n_hid=self.CONFIG.decoder_hidden,
                do_prob=self.CONFIG.decoder_dropout,
                ).double()

        if self.CONFIG.optimizer == "Adam":
            optimizer = optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()), lr=self.CONFIG.lr
            )
        elif self.CONFIG.optimizer == "LBFGS":
            optimizer = optim.LBFGS(
                list(encoder.parameters()) + list(decoder.parameters()), lr=self.CONFIG.lr
            )
        elif self.CONFIG.optimizer == "SGD":
            optimizer = optim.SGD(
                list(encoder.parameters()) + list(decoder.parameters()), lr=self.CONFIG.lr
            )

        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.CONFIG.lr_decay, gamma=self.CONFIG.gamma
        )

        # Linear indices of an upper triangular mx, used for acc calculation
        triu_indices = get_triu_offdiag_indices(self.CONFIG.data_variable_size)
        tril_indices = get_tril_offdiag_indices(self.CONFIG.data_variable_size)

        if self.CONFIG.prior:
            prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
            print("Using prior")
            print(prior)
            log_prior = torch.DoubleTensor(np.log(prior))
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = Variable(log_prior)

            if self.CONFIG.cuda:
                log_prior = log_prior.cuda()

        if self.CONFIG.cuda:
            encoder.cuda()
            decoder.cuda()
            rel_rec = rel_rec.cuda()
            rel_send = rel_send.cuda()
            triu_indices = triu_indices.cuda()
            tril_indices = tril_indices.cuda()

        rel_rec = Variable(rel_rec)
        rel_send = Variable(rel_send)



        prox_plus = torch.nn.Threshold(0.0, 0.0)

        # compute constraint h(A) value
        def _h_A(A, m):
            expm_A = matrix_poly(A * A, m)
            h_A = torch.trace(expm_A) - m
            return h_A

        def stau(w, tau):
            w1 = prox_plus(torch.abs(w) - tau)
            return torch.sign(w) * w1

        def update_optimizer(optimizer, original_lr, c_A):
            """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
            MAX_LR = 1e-2
            MIN_LR = 1e-4

            estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
            if estimated_lr > MAX_LR:
                lr = MAX_LR
            elif estimated_lr < MIN_LR:
                lr = MIN_LR
            else:
                lr = estimated_lr

            # set LR
            for parame_group in optimizer.param_groups:
                parame_group["lr"] = lr

            return optimizer, lr


        # ===================================
        # training:
        # ===================================


        def train(CONFIG, epoch, best_val_loss, lambda_A, c_A, optimizer):
            t = time.time()
            nll_train = []
            kl_train = []
            mse_train = []
            shd_trian = []

            encoder.train()
            decoder.train()
            scheduler.step()

            # update optimizer
            optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

            for batch_idx, (data, relations) in enumerate(train_loader):

                if CONFIG.cuda:
                    data, relations = data.cuda(), relations.cuda()
                data, relations = Variable(data).double(), Variable(relations).double()

                # reshape data
                relations = relations.unsqueeze(2)

                optimizer.zero_grad()

                (
                    enc_x,
                    logits,
                    origin_A,
                    adj_A_tilt_encoder,
                    z_gap,
                    z_positive,
                    myA,
                    Wa,
                ) = encoder(
                    data, rel_rec, rel_send
                )  # logits is of size: [num_sims, z_dims]
                edges = logits

                dec_x, output, adj_A_tilt_decoder = decoder(
                    data,
                    edges,
                    CONFIG.data_variable_size * CONFIG.x_dims,
                    rel_rec,
                    rel_send,
                    origin_A,
                    adj_A_tilt_encoder,
                    Wa,
                    )

                if torch.sum(output != output):
                    print("nan error\n")

                target = data
                preds = output
                variance = 0.0

                # reconstruction accuracy loss
                loss_nll = nll_gaussian(preds, target, variance)

                # KL loss
                loss_kl = kl_gaussian_sem(logits)

                # ELBO loss:
                loss = loss_kl + loss_nll

                # add A loss
                one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
                sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

                # other loss term
                if CONFIG.use_A_connect_loss:
                    connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                    loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

                if CONFIG.use_A_positiver_loss:
                    positive_gap = A_positive_loss(one_adj_A, z_positive)
                    loss += 0.1 * (
                            lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap
                    )

                # compute h(A)
                h_A = _h_A(origin_A, CONFIG.data_variable_size)
                loss += (
                        lambda_A * h_A
                        + 0.5 * c_A * h_A * h_A
                        + 100.0 * torch.trace(origin_A * origin_A)
                        + sparse_loss
                )  # +  0.01 * torch.sum(variance * variance)

                loss.backward()
                loss = optimizer.step()

                myA.data = stau(myA.data, CONFIG.tau_A * lr)

                if torch.sum(origin_A != origin_A):
                    print("nan error\n")

                # compute metrics
                graph = origin_A.data.clone().numpy()
                graph[np.abs(graph) < CONFIG.graph_threshold] = 0

                mse_train.append(F.mse_loss(preds, target).item())
                nll_train.append(loss_nll.item())
                kl_train.append(loss_kl.item())

            # print(h_A.item())
            nll_val = []
            acc_val = []
            kl_val = []
            mse_val = []

            # print(
            #     "Epoch: {:04d}".format(epoch),
            #     "nll_train: {:.10f}".format(np.mean(nll_train)),
            #     "kl_train: {:.10f}".format(np.mean(kl_train)),
            #     "ELBO_loss: {:.10f}".format(np.mean(kl_train) + np.mean(nll_train)),
            #     "mse_train: {:.10f}".format(np.mean(mse_train)),
            #     "time: {:.4f}s".format(time.time() - t),
            # )

            if "graph" not in vars():
                print("error on assign")

            return (
                np.mean(np.mean(kl_train) + np.mean(nll_train)),
                np.mean(nll_train),
                np.mean(mse_train),
                graph,
                origin_A,
            )

        t_total = time.time()
        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_epoch = 0
        best_ELBO_graph = []
        best_NLL_graph = []
        best_MSE_graph = []
        # optimizer step on hyparameters
        c_A = self.CONFIG.c_A
        lambda_A = self.CONFIG.lambda_A
        h_A_new = torch.tensor(1.0)
        h_tol = self.CONFIG.h_tol
        k_max_iter = int(self.CONFIG.k_max_iter)
        h_A_old = np.inf

        try:
            for step_k in range(k_max_iter):
                while c_A < 1e20:
                    for epoch in range(self.CONFIG.epochs):
                        ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(
                            self.CONFIG, epoch, best_ELBO_loss, lambda_A, c_A, optimizer
                        )
                        if ELBO_loss < best_ELBO_loss:
                            best_ELBO_loss = ELBO_loss
                            best_epoch = epoch
                            best_ELBO_graph = graph

                        if NLL_loss < best_NLL_loss:
                            best_NLL_loss = NLL_loss
                            best_epoch = epoch
                            best_NLL_graph = graph

                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch
                            best_MSE_graph = graph

                    # print("Optimization Finished!")
                    # print("Best Epoch: {:04d}".format(best_epoch))
                    print(
                        "nll_train: {:.10f}".format(np.mean(NLL_loss)),
                        "ELBO_loss: {:.10f}".format(np.mean(ELBO_loss)),
                        "mse_train: {:.10f}".format(np.mean(MSE_loss)),
                        # "time: {:.4f}s".format(time.time() - t),
                    )
                    if ELBO_loss > 2 * best_ELBO_loss:
                        break

                    # update parameters
                    A_new = origin_A.data.clone()
                    h_A_new = _h_A(A_new, self.CONFIG.data_variable_size)
                    if h_A_new.item() > 0.25 * h_A_old:
                        c_A *= 10
                    else:
                        break

                    # update parameters
                    # h_A, adj_A are computed in loss anyway, so no need to store
                h_A_old = h_A_new.item()
                lambda_A += c_A * h_A_new.item()

                if h_A_new.item() <= h_tol:
                    break

        except KeyboardInterrupt:
            print("Done!")

        # Create binary adjacency matrix using config threshold
        matG1 = np.matrix(origin_A.data.clone().numpy())
        final_df = pd.DataFrame(matG1, index=self.CONFIG.column_names, columns=self.CONFIG.column_names)

        for column in self.CONFIG.column_names:
            final_df[column] = np.where(np.abs(final_df[column]) < self.CONFIG.graph_threshold, 0, 1)

        # Save final binary adjacency matrix
        # final_df.to_csv("dense_adjacency_matrix.csv", index=True)

        adjacent_matrix = np.array(final_df)
        edge_table = self._get_edge_table(method_name=self.CONFIG.encoder+'-'+self.CONFIG.decoder, is_sparse=False, edge_data_map=adjacent_matrix)


        return edge_table
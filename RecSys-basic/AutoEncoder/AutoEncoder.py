import os
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, model_conf, num_users, num_items, device):
        super(AutoEncoder, self).__init__()
        self.data_name = model_conf.data_name
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.hidden_neuron = model_conf.hidden_neuron
        self.lambda_value = model_conf.lambda_value

        self.encoder = nn.Linear(self.num_items, self.hidden_neuron)
        self.decoder = nn.Linear(self.hidden_neuron, self.num_items)

        self.to(self.device)

    def forward(self, x):
        x = F.dropout(x, 0.5, training=self.training)
        enc = self.encoder(x)
        enc = torch.sigmoid(enc)

        dec = self.decoder(enc)

        return torch.sigmoid(dec)

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        self.train()
        # user, item, rating pairs
        train_matrix = dataset.train_matrix
        mask_train_matrix = dataset.org_train_matrix

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
            batch_mask_matrix = torch.FloatTensor(mask_train_matrix[batch_idx].toarray()).to(self.device)
            pred_matrix = self.forward(batch_matrix)

            pre_rec_cost = torch.mul((batch_matrix - pred_matrix), batch_mask_matrix)
            rec_cost = self.l2_norm(pre_rec_cost)
            pre_reg_cost = self.l2_norm(self.encoder.weight) + self.l2_norm(self.decoder.weight)
            reg_cost = self.lambda_value * 0.5 * pre_reg_cost

            batch_loss = rec_cost + reg_cost

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def l2_norm(self, w):
        return torch.square(torch.sqrt(torch.sum(torch.square(w))))

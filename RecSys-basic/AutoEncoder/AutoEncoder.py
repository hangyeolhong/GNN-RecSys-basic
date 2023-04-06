import os
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, data_name, num_users, num_items, hid_dim, device):
        super(AutoEncoder, self).__init__()
        self.data_name = data_name
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.hid_dim = hid_dim

        self.encoder = nn.Linear(self.num_items, self.hid_dim)
        self.decoder = nn.Linear(self.hid_dim, self.num_items)

        self.to(self.device)

    def l2_norm(self, w):
        return torch.square(torch.sqrt(torch.sum(torch.square(w))))

    def forward(self, x):
        x = F.dropout(x, 0.5, training=self.training)
        enc = self.encoder(x)
        enc = torch.sigmoid(enc)

        dec = self.decoder(enc)

        return torch.sigmoid(dec)

    def train(self, dataset, optimizer, batch_size, verbose):
        self.train()

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

            lambda_v = 0.01
            reg_cost = lambda_v * 0.5 * pre_reg_cost

            batch_loss = rec_cost + reg_cost

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

import os
import math
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, data_name, num_users, num_items, emb_dim, device):
        super(LightGCN, self).__init__()
        self.item_embedding_pred = None
        self.user_embedding_pred = None
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.data_name = data_name
        self.num_users = num_users
        self.num_items = num_items

        self.emb_dim = emb_dim
        self.num_layers = 2

        self.reg = 0.0001
        self.batch_size = 4096

        self.Graph = None
        self.data_loader = None

        self.path = "graph"
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        # weight initalization
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)  # (Tensor, mean, std)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        self.to(self.device)

    def get_emb(self, graph):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        embs = [all_emb]

        for k in range(self.num_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        lightgcn_out = torch.mean(embs, dim=1)
        # print(lightgcn_out, lightgcn_out.shape, self.num_users, self.num_items)
        users, items = torch.split(lightgcn_out, [self.num_users, self.num_items])  # num_users ~ num_users + num_items

        return users, items

    def forward(self, user, pos, neg):
        u, i = self.get_emb(self.Graph)

        user_latent = F.embedding(user, u)
        positive_latent = F.embedding(pos, i)

        positive_score = torch.mul(user_latent, positive_latent).sum(1)

        if neg is not None:
            negative_latent = F.embedding(neg, i)
            negative_score = torch.mul(user_latent, negative_latent).sum(1)
            return positive_score, negative_score
        else:
            return positive_score

    def train(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix

        self.data_loader = PairwiseGenerator(train_matrix,
                                             num_negatives=1,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             device=self.device)

        loss = 0.0
        for idx, batch_data in enumerate(self.data_loader):
            optimizer.zero_grad()
            batch_user, batch_pos, batch_neg = batch_data

            pos_output, neg_output = self.forward(batch_user, batch_pos, batch_neg)
            user_emb = self.user_embedding(batch_user)
            pos_emb = self.item_embedding(batch_pos)
            neg_emb = self.item_embedding(batch_neg)

            batch_loss = -F.sigmoid(pos_output - neg_output).log().mean()
            reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(
                len(batch_user))

            batch_loss = batch_loss + self.reg * reg_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        return loss


class PairwiseGenerator:
    def __init__(self, matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()

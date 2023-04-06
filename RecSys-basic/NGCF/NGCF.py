import os
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, data_name, num_user, num_item, emb_dim, device):
        super(NGCF, self).__init__()

        self.weight_dict = nn.ParameterDict()
        self.item_embedding_pred = None
        self.user_embedding_pred = None
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.data_name = data_name
        self.num_users = num_user
        self.num_items = num_item
        self.emb_dim = emb_dim
        self.num_layers = 2
        self.mess_dropout = 0.1

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
        # Init the weight of user-item
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        layers = [self.emb_dim] * (self.num_layers + 1)

        for k in range(len(layers) - 1):
            # W matrix and bias for graph convolution
            self.weight_dict.update(
                {'W_gc_%d' % k: nn.Parameter(nn.init.normal_(torch.empty(layers[k], layers[k + 1])))})
            self.weight_dict.update({'b_gc_%d' % k: nn.Parameter(nn.init.normal_(torch.empty(1, layers[k + 1])))})

            # W matrix and bias for bi messages of neighbors
            self.weight_dict.update(
                {'W_bi_%d' % k: nn.Parameter(nn.init.normal_(torch.empty(layers[k], layers[k + 1])))})
            self.weight_dict.update({'b_bi_%d' % k: nn.Parameter(nn.init.normal_(torch.empty(1, layers[k + 1])))})

        self.to(self.device)

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

            pos_out, neg_out = self.forward(batch_user, batch_pos, batch_neg)
            user_emb = self.user_embedding(batch_user)
            pos_emb = self.item_embedding(batch_pos)
            neg_emb = self.item_embedding(batch_neg)

            batch_loss = torch.mean(F.softplus(neg_out - pos_out))
            reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(
                len(batch_user))

            batch_loss = batch_loss + self.reg * reg_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and idx % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (idx, num_batches, batch_loss))
        return loss

    def forward(self, user, pos, neg=None):
        u, i = self.get_emb(self.Graph)

        user_latent = F.embedding(user, u)
        pos_latent = F.embedding(pos, i)

        pos_score = torch.mul(user_latent, pos_latent).sum(1)
        if neg is not None:
            # BPR loss
            neg_latent = F.embedding(neg, i)
            neg_score = torch.mul(user_latent, neg_latent).sum(1)
            return pos_score, neg_score
        else:
            return pos_score

    def get_emb(self, graph):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb])
        print(all_emb.shape)

        embs = [all_emb]
        ego_emb = all_emb  # (N + M) x d

        for k in range(self.num_layers):
            side_emb = torch.sparse.mm(graph, ego_emb)  # ((N + M) x (N + M)) * ((N + M) x d) = (N + M) x d

            # transformed sum messages of neighbors
            sum_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % k]) + self.weight_dict[
                'b_gc_%d' % k]  # (N + M) x d'

            # element-wise product
            bi_emb = torch.mul(ego_emb, side_emb)  # (N + M) x d
            # transformed bi messages of neighbors
            bi_emb = torch.matmul(bi_emb, self.weight_dict['W_bi_%d' % k]) + self.weight_dict[
                'b_bi_%d' % k]  # (N + M) x d'

            # non-linear activation
            ego_emb = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)
            # message dropout
            ego_emb = F.dropout(ego_emb, self.mess_dropout, training=self.training)

            # normalize the distribution of embeddings
            norm_emb = F.normalize(ego_emb, p=2, dim=1)
            embs += [norm_emb]

        embs = torch.stack(embs, dim=1)

        ngcf_out = torch.mean(embs, dim=1)
        users, items = torch.split(ngcf_out, [self.num_users, self.num_items])
        return users, items


class PairwiseGenerator:
    def __init__(self, matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()

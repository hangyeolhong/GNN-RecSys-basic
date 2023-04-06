import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, model_conf, num_users, num_items, device):
        super(MF, self).__init__()

        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.data_name = model_conf.data_name
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = model_conf.hidden_dim
        self.batch_size = model_conf.batch_size
        self.reg = model_conf.reg
        self.pointwise = model_conf.pointwise

        self.device = device

        self.data_loader = None
        self.user_embedding_pred = None
        self.item_embedding_pred = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)  # (tensor, mean, std)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        self.to(self.device)

    def forward(self, user, pos, neg):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        user_latent = F.embedding(user, user_emb)

        # positive
        positive_latent = F.embedding(pos, item_emb)
        positive_score = torch.sum(torch.mul(user_latent, positive_latent), 1)  # element-wise product

        if not self.pointwise:
            # negative
            negative_latent = F.embedding(neg, item_emb)
            negative_score = torch.sum(torch.mul(user_latent, negative_latent), 1)

            return positive_score, negative_score

        else:
            return positive_score, None

    def train(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix

        if self.pointwise:
            self.data_loader = PointwiseGenerator(train_matrix,
                                                  num_negatives=1,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  device=self.device)
        else:
            self.data_loader = PairwiseGenerator(train_matrix,
                                                 num_negatives=1,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 device=self.device)

        loss = 0.0
        num_batches = len(self.data_loader)
        for idx, batch_data in enumerate(self.data_loader):
            optimizer.zero_grad()
            batch_user, batch_pos, batch_neg = batch_data
            # if self.pointwise: batch_neg = [1] * len(batch_pos)
            # if not self.pointwise: batch_neg = [item #] * len(batch_neg)

            pos_output, neg_output = self.forward(batch_user, batch_pos, batch_neg)
            userEmb0 = self.user_embedding(batch_user)
            posEmb0 = self.item_embedding(batch_pos)

            if self.pointwise:
                batch_loss = F.binary_cross_entropy_with_logits(pos_output, batch_neg)
            else:
                # pairwise BPR loss
                batch_loss = -F.sigmoid(pos_output - neg_output).log().mean()

            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2))

            if not self.pointwise:
                negEmb0 = self.item_embedding(batch_neg)
                reg_loss += negEmb0.norm(2).pow(2)

            batch_loss = batch_loss + self.reg * (reg_loss / float(len(batch_user)))

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and idx % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (idx, num_batches, batch_loss))

        return loss


class PairwiseGenerator:
    def __init__(self, matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()


class PointwiseGenerator:
    def __init__(self, matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()

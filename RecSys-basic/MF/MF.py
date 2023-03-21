import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel

class MF(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(MF, self).__init__()

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

        self.build_graph()

    def build_graph(self):
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)

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

        if self.pointwise == False:
            # negative
            negative_latent = F.embedding(neg, item_emb)
            negative_score = torch.sum(torch.mul(user_latent, negative_latent), 1)

            return positive_score, negative_score

        else:
            return positive_score, None

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
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

            if self.pointwise == False:
                negEmb0 = self.item_embedding(batch_neg)
                reg_loss += negEmb0.norm(2).pow(2)

            batch_loss = batch_loss + self.reg * (reg_loss / float(len(batch_user)))

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and idx % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (idx, num_batches, batch_loss))

        return loss

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for index in range(num_batches):
                if (index + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[index * test_batch_size:]
                else:
                    batch_idx = perm[index * test_batch_size: (index + 1) * test_batch_size]

                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users).to(self.device)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()

        pred_matrix[eval_pos.nonzero()] = float('-inf')

        return pred_matrix

    def before_evaluate(self):
        self.user_embedding_pred, self.item_embedding_pred = self.user_embedding.weight, self.item_embedding.weight

    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding(user_ids)
        all_item_latent = self.item_embedding.weight.data
        return user_latent @ all_item_latent.T

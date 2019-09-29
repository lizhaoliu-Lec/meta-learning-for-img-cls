import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightGenerator(object):
    def __init__(self, base_cls=64, way=5,
                 hidden_dim=1000,
                 att_gamma=10.0):
        """Without forgetting weight generator class initializer.
        Args:
            base_cls: the number of base category
            way: the number of few shot way
            hidden_dim: feature embedding size
            att_gamma: a learnable scalar parameter γ in order to
                increase the peakiness of the softmax distribution.
        """
        self.base_cls = base_cls
        self.way = way
        self.hidden_dim = hidden_dim

        # Translate the feature vector to query vector
        # i.e., phi_q * z_i
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.query_layer.weight.data.copy_(
            torch.eye(hidden_dim, hidden_dim) +
            torch.randn(hidden_dim, hidden_dim) * 0.001)
        self.query_layer.bias.data.zero_()

        self.att_gamma = nn.Parameter(
            torch.FloatTensor(1).fill_(att_gamma), requires_grad=True)
        key = torch.FloatTensor(base_cls, hidden_dim).normal_(0.0, np.sqrt(2.0 / hidden_dim))
        self.key = nn.Parameter(key, requires_grad=True)

    def hardmard_product(self, weight):
        phi = torch.FloatTensor(self.hidden_dim).fill_(1)
        phi = nn.Parameter(phi, requires_grad=True)
        assert (weight.dim() == 2 and weight.size(1) == phi.size(0))
        out = weight * phi.expand_as(weight)
        return out

    def get_att_weight(self, embedding_shot, base_weight):
        """Get attention weight.
        Args:
            embedding_shot: data_shot after embedding, tensor of shape
                [way_x_shot, embedding_size]. NOTICE that it should be
                well-structured i.e., S = mod(way) should belong to the
                same category.
            base_weight: weight of base(i.e., pre-train) classifier
                [base_cls, embedding_size].
        Returns:
            att_weight: [way, embedding_size]
        """
        if base_weight.size(0) != self.base_cls or \
                base_weight.size(1) != self.hidden_dim:
            raise ValueError("base weight shape should be (%d, %d), but found (%d, %d)." %
                             (self.base_cls, self.hidden_dim, base_weight(0), base_weight(1)))
        # l2-norm
        embedding_shot = F.normalize(embedding_shot, p=2, dim=embedding_shot.dim() - 1)

        way_x_shot, embedding_size = embedding_shot.size()
        way, shot = self.way, way_x_shot // self.way

        # rearrange the sequence order of embedding shot
        idx = [start + i * way for start in range(way) for i in range(shot)]
        # [way_x_shot, embedding_size]
        trans_embedding_shot = embedding_shot[idx]

        # translate the feature vector into query vector
        # [way_x_shot, embedding_size]
        query_vector = self.query_layer(trans_embedding_shot)
        # [way, num_shot, embedding_size]
        query_vector = query_vector.view(way, shot, embedding_size)
        query_vector = F.normalize(query_vector, p=2, dim=query_vector.dim() - 1)

        # [way, base_cls, embedding_size]
        key_vector = torch.stack([self.key for _ in range(way)])
        # [way, embedding_size, base_cls]
        key_vector = key_vector.transpose(1, 2)
        key_vector = F.normalize(key_vector, p=2, dim=key_vector.dim() - 1)

        # [way, num_shot, base_cls]
        att_coef = self.att_gamma * torch.bmm(query_vector, key_vector)
        att_coef = F.softmax(att_coef.view(way * shot, self.base_cls))
        # [way, num_shot, base_cls]
        att_coef = att_coef.view(way, shot, self.base_cls)

        # [way, base_cls, embedding_size]
        base_weight_repeat = torch.stack([base_weight for _ in range(way)])
        base_weight_repeat = F.normalize(base_weight_repeat, p=2, dim=base_weight_repeat.dim() - 1)

        # [way, num_shot, embedding_size]
        att_weight = torch.bmm(att_coef, base_weight_repeat)
        # [way, embedding_size]
        att_weight = att_weight.mean(dim=1, keepdim=False)
        # perform Hardmard product on att_weight
        # [way, embedding_size]
        att_weight = self.hardmard_product(att_weight)

        return att_weight

    def get_avg_weight(self, embedding_shot):
        """Get average weight.
        Args:
            embedding_shot: data_shot after embedding, tensor of shape
                [way_x_shot, embedding_size]. NOTICE that it should be
                well-structured i.e., S = mod(way) should belong to the
                same category.
        Returns:
            att_weight: [way, embedding_size]
        """
        # l2-norm
        embedding_shot = F.normalize(embedding_shot, p=2, dim=embedding_shot.dim() - 1)

        way_x_shot, embedding_size = embedding_shot.size()
        way, shot = self.way, way_x_shot // self.way

        # rearrange the sequence order of embedding shot
        idx = [start + i * way for start in range(way) for i in range(shot)]
        # [way_x_shot, embedding_size]
        trans_embedding_shot = embedding_shot[idx]

        # [way, num_shot, embedding_size]
        avg_weight = trans_embedding_shot.view(way, shot, embedding_size)
        avg_weight = F.normalize(avg_weight, p=2,
                                 dim=avg_weight.dim() - 1,
                                 eps=1e-12)
        # [way, embedding_size]
        avg_weight = avg_weight.mean(dim=1, keepdim=False)
        # perform Hardmard product on avg_weight
        # [way, embedding_size]
        avg_weight = self.hardmard_product(avg_weight)

        return avg_weight

    def get_novel_weight(self, embedding_shot, base_weight):
        """Get novel weight, it is the sum of att_weight and avg_weight.
        Args:
            embedding_shot: data_shot after embedding, tensor of shape
                [way_x_shot, embedding_size]. NOTICE that it should be
                well-structured i.e., S = mod(way) should belong to the
                same category.
            base_weight: weight of base(i.e., pre-train) classifier
                [base_cls, embedding_size].
        Returns:
            novel_weight: shape of [way, embedding_size]
        """
        att_weight = self.get_att_weight(embedding_shot, base_weight)
        avg_weight = self.get_avg_weight(embedding_shot)
        novel_weight = att_weight + avg_weight
        # [way, embedding_size]
        novel_weight = novel_weight.view(self.way, self.hidden_dim)

        return novel_weight

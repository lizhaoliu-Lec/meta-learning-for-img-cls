""" Sampler for dataloader. """
import torch
import numpy as np

__all__ = ['CategoriesSampler']


class CategoriesSampler(object):
    """The class to generate episodic data

    There are total num_val_classes(16) in validation set.
    The sampler works in the following step.
    (1) group the same class together, e.g.
        self.m_id length is num_val_classes, and self.m_id[c] is all
        the labels for class c.
    (2) for each batch, randomly select n_cls (num_way) classes in self.m_id
    (3) for each class, randomly select n_per (shot + query) for each class get in (2)


    """

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                c_all = self.m_ind[c]
                pos = torch.randperm(len(c_all))[:self.n_per]
                batch.append(c_all[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # the batch has
            # n_cls * n_per
            # n_per = shot + val_query
            # p = shot * n_cls
            # :p = 0 : shot*cls
            # p: = shot*n_cls : (shot+val_query)*n_cls
            yield batch

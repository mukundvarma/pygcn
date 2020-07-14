import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init_scheme="xavier"):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init_scheme=init_scheme)
        self.gc2 = GraphConvolution(nhid, nclass, init_scheme=init_scheme)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

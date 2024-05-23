import torch
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, GCN2Conv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
import numpy as np
from torch_geometric.utils import dropout_adj

class LogReg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(2 * in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
    
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
class GCN(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight=None, K=1):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x) 
        for k in range(K):
            x = self.propagate(edge_index, x=x, norm=norm)   
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    
class Prop(MessagePassing):
    def __init__(self, num_hidden, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_hidden, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()


class Model(torch.nn.Module):
    def __init__(self, num_features, num_hidden: int, tau1: float, tau2: float, l1: float, l2: float):
        super(Model, self).__init__()
        self.tau1: float = tau1
        self.tau2: float = tau2
        self.gcn_s1 = GCN(num_features, num_hidden)
        self.gcn_s2 = GCN(num_features, num_hidden)
        self.gcn_f1 = GCN(num_features, num_hidden)
        self.gcn_f2 = GCN(num_features, num_hidden)
        self.lin1 = nn.Linear(num_features, num_hidden)
        self.prop = Prop(num_hidden, 10)
        self.linear = nn.Linear(num_hidden, 1)
        self.l1 = l1
        self.l2 = l2
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.gcn_s1.reset_parameters()
        self.gcn_s2.reset_parameters()
        self.gcn_f1.reset_parameters()
        self.gcn_f2.reset_parameters()
        self.lin1.reset_parameters()
        self.prop.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, knn_graph: torch.Tensor) -> torch.Tensor:
        
        h0 = self.gcn_s1(x, edge_index, K=2) 
        h1 = self.lin1(x)
        h1 = self.prop(h1, edge_index)
   
        
        z0 = self.gcn_f2(x, edge_index, K=2)
        w = self.linear(z0)
        z1 = self.gcn_f2(x, edge_index, K=2) *w + self.gcn_f2(x, knn_graph, K=1)*(1-w)
        
        return h0, h1, z0, z1

    def embed(self, x: torch.Tensor,
                edge_index: torch.Tensor, knn_graph: torch.Tensor) -> torch.Tensor:
        
        h0 = self.gcn_s1(x, edge_index, K=2) 
        h1 = self.lin1(x)
        h1 = self.prop(h1, edge_index)
   
        
        z0 = self.gcn_f2(x, edge_index, K=2)
        w = self.linear(z0)
        z1 = self.gcn_f2(x, edge_index, K=2) *w + self.gcn_f2(x, knn_graph, K=1)*(1-w)
        
        return (h0 + h1 + z1).detach()

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau1)
        between_sim = f(self.sim(z1, z2))
        refl_sim1 = f(self.sim(z1, z1))
        refl_sim2 = f(self.sim(z2, z2))
  
        return (-torch.log(
            between_sim.diag()
            / (refl_sim1.sum(1) + between_sim.sum(1) - refl_sim1.diag() + refl_sim2.sum(1) - refl_sim2.diag()))).mean()


    def loss(self, z0, z1, h0, h1):
        l1 = self.semi_loss(h0, h1) 
        l2 = self.semi_loss(z0, z1)
        return self.l1 * l1 + l2 


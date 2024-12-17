from torch_geometric.nn.conv import MessagePassing
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

class NGCFConv(MessagePassing):
  def __init__(self, in_size, out_size, dropout):  
    super(NGCFConv, self).__init__(aggr='add')

    self.W1 = torch.nn.Linear(in_size, out_size, bias=True)
    self.W2 = torch.nn.Linear(in_size, out_size, bias=True)

    self.leaky_relu = torch.nn.LeakyReLU(0.2)

    self.dropout = torch.nn.Dropout(dropout)
    self.init_parameters()



  def init_parameters(self):
    torch.nn.init.xavier_uniform_(self.W1.weight)
    torch.nn.init.constant_(self.W1.bias, 0)
    torch.nn.init.xavier_uniform_(self.W2.weight)
    torch.nn.init.constant_(self.W2.bias, 0)


  def forward(self, x, edge_index):
    # Compute normalization
    src, dst = edge_index
    deg = degree(src, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    # Start propagating messages
    out = self.propagate(edge_index, x=x, norm=norm)

    # Perform update after aggregation
    out += self.W1(x)
    out = self.dropout(out)
    return self.leaky_relu(out)


  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.W1(x_j) + self.W2(x_j * x_i))
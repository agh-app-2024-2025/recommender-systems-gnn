from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index

from src.models.layers.ngcf_conv import NGCFConv
from src.models.model import Model, BPRLoss


class NGCF(Model):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        alpha: Optional[Union[float, Tensor]] = None,
        init_method: str = 'xavier',
        features_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)

        self.embedding = Embedding(num_nodes, embedding_dim)
        in_size = features_dim if features_dim is not None else embedding_dim
        out_size = features_dim if features_dim is not None else embedding_dim
        self.convs = ModuleList([NGCFConv(in_size=in_size, out_size=out_size, dropout=dropout, **kwargs)] +
                                [NGCFConv(in_size=out_size, out_size=out_size, dropout=dropout, **kwargs) for _ in range(num_layers-1)])
        self.init_method = init_method

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'xavier':
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif self.init_method == 'normal':
            torch.nn.init.normal_(self.embedding.weight, std=0.01)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        x: OptTensor = None,
    ) -> Tensor:
        if x is None:
            x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
        self,
        edge_index: Adj,
        x: OptTensor = None,
        edge_label_index: OptTensor = None,
    ) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, x)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
    ) -> Tensor:
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        input_x: OptTensor = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.embedding.weight if input_x is None else input_x
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

from torch_geometric.utils import degree
import torch
import tqdm
from typing import Tuple

@torch.no_grad()
def test(model, data, train_edge_label_index, num_users, k: int, batch_size: int, use_node_features):
    if use_node_features:
        input_x = data.x
    else:
        input_x = None
    emb = model.get_embedding(data.edge_index, input_x)
    num_test_users = max(data.edge_label_index[0]) + 1
    user_emb, item_emb = emb[:num_users], emb[num_users:]

    precision = recall = hits = total_examples = 0
    for start in range(0, num_test_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ item_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision, recall, and hits@k:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        hits += float((isin_mat.sum(dim=-1) > 0).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples, hits / total_examples

@torch.no_grad()
def test_kg(
    model,
    data,
    train_edge_label_index,
    batch_size: int,
    num_users: int,
    num_items: int,
    k: int = 10,
    log: bool = True,
) -> Tuple[float, float, float]:
    head_index, tail_index = data.edge_index[0], data.edge_index[1]
    device=head_index.device
    rel_type = torch.tensor(data.edge_type[0], device=device) # only one relation type to test- user item
    num_test_users = max(head_index) + 1

    arange = range(num_test_users)
    arange = tqdm(arange) if log else arange

    precision = recall = total_hit = 0
    for h in arange:
        scores = []
        tail_indices = torch.arange(num_users, num_users + num_items, device=device)

        for ts in tail_indices.split(batch_size):
            scores.append(model(torch.tensor(h, device=device).expand_as(ts), rel_type.expand_as(ts), ts))
        scores = torch.cat(scores)
        label_index = tail_index[head_index == h] - num_users
        if len(label_index) == 0: # no test data for this user
            num_test_users -=1
            continue

        train_label_index = train_edge_label_index[1][train_edge_label_index[0] == h] - num_users
        train_label_index = train_label_index[train_label_index < num_items]
        scores[train_label_index] = float('-inf')
        topk_index = scores.topk(k, dim=-1).indices

        num_hits = torch.isin(label_index, topk_index).sum().item()
        total_hit += num_hits > 0
        precision += num_hits / k
        recall += num_hits / len(label_index)

    return precision / num_test_users, recall / num_test_users, total_hit / num_test_users

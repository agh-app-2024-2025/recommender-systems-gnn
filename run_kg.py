import argparse
import os.path as osp
import time

import torch
import torch.optim as optim

from src.dataloading.kg_dataset import KGMovieLens100K
from src.training.test_utils import test_kg
from src.utils import save_metrics_to_file
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}

model_arg_map = {'rotate': {'margin': 9.0}}

def train(model, loader, optimizer):
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(model, data, train_edge_index, num_users, num_items):
    model.eval()
    return test_kg(model, data, train_edge_index, batch_size=1000, num_users=num_users, num_items=num_items,
                   k=20, log=False)

def main(model_name, batch_size, num_epochs, use_only_user_item, hidden_channels):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = KGMovieLens100K(split='train', use_only_item_user=use_only_user_item)
    num_users, num_items = train_dataset.num_users, train_dataset.num_items

    train_data = train_dataset[0].to(device)
    test_data = KGMovieLens100K(split='test')[0].to(device)

    model = model_map[model_name](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=hidden_channels,
    **model_arg_map.get(model_name, {}),
    ).to(device)

    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
    }
    optimizer = optimizer_map[model_name]   

    loss_list, precision_list, recall_list, hits_list = [], [], [], []

    start = time.time()
    for epoch in range(num_epochs):
        loss = train(model, loader, optimizer)
        precision, recall, hits = test(model, test_data, train_data.edge_index, num_users, num_items)
        loss_list.append(loss)
        precision_list.append(precision.item())
        recall_list.append(recall.item())
        hits_list.append(hits.item())
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
            f'{precision:.4f}, Recall@20: {recall:.4f}, HR@20: {hits:.4f}')
    end = time.time()
    training_time = end - start
        
    args = [model, batch_size, num_epochs, use_only_user_item]
    save_metrics_to_file('kg', training_time, args, loss_list, precision_list, recall_list, hits_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommender System using KG')
    parser.add_argument('--model', choices=model_map.keys(), type=str.lower, default='transe', help='Model name')
    parser.add_argument('--batch_size', type=int, default=8000, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_only_user_item', action='store_true', help='Whether to use only user-item edges in the model')
    parser.add_argument('--hidden_channels', type=int, default=50, help='Hidden channels dimension')
    args = parser.parse_args()
    main(args.model, args.batch_size, args.num_epochs, args.use_only_user_item, args.hidden_channels)
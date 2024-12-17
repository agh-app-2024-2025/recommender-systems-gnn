import torch
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.training.train_utils import train
from src.training.test_utils import test
from src.dataloading.dataset import Dataset
from src.utils import save_metrics_to_file
import argparse
import time



def main(dataset_name, batch_size, num_layers, embedding_dim, num_epochs, init_method, model_name, use_node_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(dataset_name)
    dataset.data = dataset.data.to(device)

    train_edge_label_index = dataset.train_edge_label_index
    num_users, num_items, num_nodes = dataset.num_users, dataset.num_items, dataset.num_nodes
    features_dim = dataset.data.x.size(1) if use_node_features else None
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )

    if model_name == 'LightGCN':
        if use_node_features:
            raise ValueError("LightGCN does not support node features")
        model = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            init_method=init_method,
        ).to(device)
    elif model_name == 'NGCF':
        model = NGCF(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            init_method=init_method,
            features_dim=features_dim
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    loss_list, precision_list, recall_list, hits_list = [], [], [], []

    start = time.time()
    for epoch in range(num_epochs):
        loss = train(model, dataset.data, train_loader, train_edge_label_index, num_users, num_items, use_node_features, device)
        precision, recall, hits = test(model, dataset.data, train_edge_label_index, num_users, 20, batch_size, use_node_features)
        loss_list.append(loss)
        precision_list.append(precision)
        recall_list.append(recall)
        hits_list.append(hits)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
            f'{precision:.4f}, Recall@20: {recall:.4f}, HR@20: {hits:.4f}')
    end = time.time()
    training_time = end - start
    args = [dataset_name, batch_size, num_layers, embedding_dim, num_epochs, init_method, model_name, use_node_features]  
    save_metrics_to_file('cf', training_time, args, loss_list, precision_list, recall_list, hits_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Recommender System using GNN')
    parser.add_argument('--dataset', type=str, choices=['MovieLens100K', 'AmazonBook'], default='MovieLens100K', help='Dataset name')
    parser.add_argument('--init_method', type=str, default='xavier', help='Initialization method for weights')
    parser.add_argument('--batch_size', type=int, default=8000, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--model', type=str, choices=['LightGCN', 'NGCF'], default='NGCF', help='Model name')
    parser.add_argument('--use_node_features', action='store_true', help='Whether to use node features in the model')
    args = parser.parse_args()
    main(args.dataset, args.batch_size, args.num_layers, args.embedding_dim, args.num_epochs, args.init_method, args.model, args.use_node_features)
    
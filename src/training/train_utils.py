import torch
from tqdm import tqdm


def train(model, data, train_loader, train_edge_label_index, num_users, num_movie, use_node_features, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movie, (index.numel(), ), device=device)
        ], dim=0)
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

        optimizer.zero_grad()
        if use_node_features:
            input_x = data.x
        else:
            input_x = None
        pos_rank, neg_rank = model(data.edge_index, input_x, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
            input_x=input_x
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples
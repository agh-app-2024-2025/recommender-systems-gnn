from torch_geometric.datasets import MovieLens100K, AmazonBook

class Dataset:
    def __init__(self, dataset_name: str, is_homo: bool = True):
        self.dataset_name = dataset_name
        if self.dataset_name == 'MovieLens100K':
            self.dataset = MovieLens100K('./data/raw/ml-100k')
        elif self.dataset_name == 'AmazonBook':
            self.dataset = AmazonBook('./data/raw/AmazonBook')
        else:
            raise NotImplementedError('Dataset not implemented')    
        self.data = self.dataset[0]
        self.num_users, self.num_items, self.num_nodes = self.get_node_numbers()
        if is_homo:
            self.order_users_before_items()
            self.data = self.data.to_homogeneous()

    def order_users_before_items(self):
        if self.dataset_name == "MovieLens100K":
            self.data.rename('movie', 'Movie')

    def get_node_numbers(self):
        if self.dataset_name == "MovieLens100K":
            return self.data['user'].num_nodes, self.data['movie'].num_nodes, self.data.num_nodes
        elif self.dataset_name == "AmazonBook":
            return self.data['user'].num_nodes, self.data['book'].num_nodes, self.data.num_nodes

    @property
    def train_edge_label_index(self):   
        mask = self.data.edge_index[0] < self.data.edge_index[1]
        return self.data.edge_index[:, mask]
# Graph Neural Networks in Recommender Systems

## How to Run the Project

Make sure you have Python installed. Follow these steps to run the recommender-systems-gnn project:

1. Navigate to the project directory:
    ```bash
    cd recommender-systems-gnn
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the collaborative filtering model remembering to set the desired parameters:
    ```bash
    python run_cf.py
    ```

4. Run the knowledge graph model remembering to set the desired parameters:
    ```bash
    python run_kg.py
    ```

## Documentation

The repository that consists of the following main folders:

- **data**: A folder for storing the dataset in raw format as well as its preprocessed versions to enable quick loading.
- **notebooks**: Machine learning model training often occurs in Jupyter Notebooks. While our models are launched from the command line, notebooks are useful for data visualization.
- **results**: To perform thorough analyses of the models, we save results in CSV files. These files contain metrics such as loss, precision, recall, and hit ratio at specific epochs, along with training time and model hyperparameters.
- **src**: This folder, short for "source code," contains the core implementation of graph neural network models.

The main part of the system is in the `src` folder. It is divided into several modules, as described below.

### Dataloading
The Dataloading module is designed to handle collaborative filtering (CF) and knowledge graph (KG) datasets. It is responsible for converting raw data into a format compatible with GNN models. Dataset classes preprocess and store information about users, items, and interactions, making it easy to feed networks with the structured data they require.

### Training
This module contains functions for model training and testing, which are similar for both CF and KG. The training loop is structured to optimize the model for better item recommendations. It iterates over batches of data provided by the training data loader. For each batch, positive and negative labels are sampled, with negative labels created by pairing positive labels with randomly selected nodes representing items. 

The model computes scores for both positive and negative samples. A recommendation loss, calculated using a previously described algorithm, is backpropagated through the model, and the optimizer is updated. Training statistics (total loss and total number of examples) are accumulated during the loop.

The testing loop iterates over test users. For each user, it computes scores for all items using the GNN model, excluding training edges to evaluate unseen data. Metrics such as Precision@K, Recall@K, and Hits@K are computed based on the top K items.

### Models
To explain the implementation of the models, we first elaborate on how user-defined PyTorch Geometric models are built. Collaborative filtering networks derive from the `torch.nn.Module` class, which acts as a container for neural network layers. GNNs, apart from standard layers like activation, normalization, or dropout, utilize graph layers, often Graph Convolutional Layers. These layers are based on the PyG `MessagePassing` class, which allows users to define custom message, propagate, and update functions for message aggregation processes.

We implemented the LightGCN and NGCF models. These models provide an interface to retrieve embeddings for given edges and recommendations for given users. The forward pass of each model considers the module layers, particularly graph layers. For LightGCN, we integrated the `LGConv` layer available in PyTorch Geometric. For NGCF, we implemented the `NGCFConv` class ourselves. 

The forward function of `NGCFConv` first computes normalization, then propagates messages along edges. Finally, after aggregation, it performs an update

For Knowledge Graphs, PyTorch Geometric provides a specialized module, `nn.kge`, where KG models such as TransE, RotatE, DistMult, and ComplEx are implemented. These models derive from a base class, `KGEModel`, which offers an interface to calculate loss, perform forward passes, and reset parameters. The primary distinction between the models lies in how the relationships depend on the nodes, a detail accounted for in each model's implementation.

### Launching
Two scripts are prepared for launching models: `run_cf.py` and `run_kg.py`. In both scripts, the training and testing processes are executed in a loop over epochs, with metrics collected and saved to a file. These scripts offer flexibility in selecting models and hyperparameters, making them ideal for experimenting with different configurations. 

To launch a script, the user can run `python3 run_cf.py` or `python3 run_kg.py` with the desired arguments. If certain arguments are not specified, default values are used.


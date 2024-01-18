import torch
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9

from tqdm import tqdm

class QM9Dataset(InMemoryDataset):
    # An extension to the QM9 dataset from PyG used for preprocessing the data for the model
    def __init__(self, root, dataset_info) -> None:
        self.info = dataset_info
        self.dataset = QM9(root=root)

        super().__init__(root)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return self.dataset.raw_file_names

    @property
    def processed_file_names(self):
        return ['custom_data.pt']

    def download(self):
        self.dataset.download()

    def process(self):
        # Converts graphs to use one-hot encoded features and remove graph labels
        print("Custom processing...")
        data_list = [self.convert(g) for g in tqdm(self.dataset)]
        # Saves processed graphs
        self.save(data_list, self.processed_paths[0])
        
    def convert(self, g):
        # Get one-hot encoding of atom types
        x = F.one_hot(torch.tensor([self.info.encode_atoms[mol.item()] for mol in g.z]), num_classes=len(self.info.encode_atoms)).float()

        # Do not use labels for this dataset
        y = torch.zeros((1, 0), dtype=torch.float)

        # Create space in edge attributes to encode no edge
        edge_attr = torch.zeros((g.edge_attr.size(0), 1))
        edge_attr = torch.cat((edge_attr, g.edge_attr), dim=1)

        # Create PyG data object representing a graph
        data = pyg.data.Data(x=x, edge_index=g.edge_index, edge_attr=edge_attr, y=y)

        return data

class QM9DatasetInfo():
    # Information regarding the training and sampling using the QM9 dataset
    def __init__(self) -> None:
        # Dimensions for transformer layer
        self.input_dims = { 'X':5, 'E':5, 'y':1 }
        self.output_dims = { 'X':5, 'E':5, 'y':1 }
        self.hidden_dims = { 'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 
                            'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128 }
        self.hidden_mlp_dims = { 'X': 256, 'E': 128, 'y': 128 }
        # Min and Max nodes for generation purposes
        self.min_nodes = 4
        self.max_nodes = 29
        # { 1:'H', 6:'C', 7:'N', 8:'O', 9:'F' }
        self.encode_atoms = { 1:0, 6:1, 7:2, 8:3, 9:4 }
        self.decode_atoms = { 0:1, 1:6, 2:7, 3:8, 4:9 }
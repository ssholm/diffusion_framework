import os
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw

from diffusion import Diffusion
from models import GraphTransformer
from datasets import *

def get_model(device, dataset_info, n_layers):
    # Setup model and diffusion process on device
    diffusion = Diffusion(device=device)
    model = GraphTransformer(n_layers, dataset_info).to(device)
    model = torch.nn.DataParallel(model)
    model.to(device)

    return diffusion, model

def get_data(data_name: str, batch_size:int):
    # Load dataset and split into training and validation sets
    if data_name == 'qm9':
        dataset_info = QM9DatasetInfo()
        dataset = QM9Dataset(root='./data/qm9', dataset_info=dataset_info)
        train_split, val_split = dataset[:100000], dataset[100000:]
    else:
        print('Unknown dataset', data_name)
        exit()

    # Split dataset into training (90%) and validation set (10%) if splitting is not done priorly
    if train_split is None or val_split is None:
        train_split, val_split = torch.utils.data.random_split(dataset=dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    # Create training and validation dataloader to iterate the dataset
    train = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_split, batch_size=20, shuffle=True)

    return train, val, dataset_info

def create_checkpoint(model, optim, save_path, epoch):
    # Save model and optimiser parameters as a checkpoint
    torch.save(model.state_dict(), os.path.join(save_path, "models", f'epoch_{epoch}_model.pt'))
    torch.save(optim.state_dict(), os.path.join(save_path, "models", f'epoch_{epoch}_optim.pt'))

def load_checkpoint(model, optim, save_path, epoch):
    # Load model and optimiser parameters from a checkpoint
    model.load_state_dict(torch.load(os.path.join(save_path, "models", f'epoch_{epoch}_model.pt')))
    optim.load_state_dict(torch.load(os.path.join(save_path, "models", f'epoch_{epoch}_optim.pt')))

    return model, optim

def save_molecules(mols, path):
    # Save molecules using RDKit
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
    img.save(path)

def prepare_graph(X, E):
    # Prepares a batch of graphs for conversion to molecules
    graphs = []
    # Iterate every graph
    for i in range(X.size(0)):
        X_i, E_i = X[i], E[i]
        # Remove unused nodes
        nodes = torch.argmin(torch.sum(X_i, dim=1)).item()
        X_i = X_i[:nodes]
        E_i = E_i[:nodes, :nodes]
        # Add converted graph to result list
        graphs.append((X_i, E_i))
        
    return graphs

def mol_from_graph(x, e, dataset_info):
    # Converts a graph into a RDKit molecule
    decode_bond = {
        0: None, 
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC
    }

    # Create raw molecule
    mol = Chem.RWMol()
    
    # Add atoms to molecule
    nodes_idx = {}
    for ia, a in enumerate(torch.argmax(x, dim=1)):
        atom = Chem.Atom(dataset_info.decode_atoms[a.item()])
        idx = mol.AddAtom(atom)
        nodes_idx[ia] = idx
    
    # Add bonds between atom in molecule
    for ix, row in enumerate(torch.argmax(e, dim=2)):
        for iy, bond in enumerate(row):
            bond = decode_bond[bond.item()]
            if iy <= ix or bond is None:
                continue
            mol.AddBond(nodes_idx[ix], nodes_idx[iy], bond)

    # Return molecule if valid otherwise return empty molecule
    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        print("Can't kekulize molecule")
        mol = None
    return mol

def setup_logging(save_path):
    # Initialises folder structure for training, evaluation and sampling
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "eval"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "test"), exist_ok=True)


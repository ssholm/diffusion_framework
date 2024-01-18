from rdkit import Chem
from tqdm import tqdm
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from utils import *
from model_utils import *


def compute_validity(mols, log=True, position=0):
    # Computes validity using RDKit
    if log: print("Computing Validity")
    valid = []
    all_smiles = []

    # Iterate every molecule
    for mol in tqdm(mols, position=position, leave=(position==0)):
        try:
            # Get largest fragment if the sample contains multiple molecules
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            # Sanitise the molecule
            Chem.SanitizeMol(largest_mol)
            # Generate unique SMILES string for the molecule and add it to lists
            smiles = Chem.MolToSmiles(largest_mol)
            valid.append(smiles)
            all_smiles.append(smiles)
        except (ValueError, Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
            all_smiles.append(None)

    return valid, all_smiles

def compute_uniqueness(smiles):
    # Input SMILES has to be list of valid SMILES
    print("Computing Uniqueness")
    # Return list of unique SMILES
    return list(set(smiles))

def compute_novelty(smiles, smiles_path):
    # Computes the list of novel samples based on SMILES from the traiing dataset
    print("Computing Novelty")
    dataset_smiles = []
    # Load training SMILES
    with open(smiles_path, 'r') as f:
        dataset_smiles = f.read().splitlines()
    # Return novel SMILES and training SMILES
    return list(set(smiles).difference(set(dataset_smiles))), dataset_smiles

def compute_dataset_smiles(dataloader, dataset_info, save_path):
    # Make a pass over the training dataset and compute its SMILES strings
    print("Computing Dataset SMILES")
    dataset_smiles = []
    # Iterate every batch of graphs in the training dataset
    for data in tqdm(dataloader):
        # Convert graph to node features and edge features
        x, e, _ = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # Convert graph to molecule representation
        mols = [mol_from_graph(x_i, e_i, dataset_info) for x_i, e_i in prepare_graph(x, e)]
        # Compute valid set of SMILES for the training dataset
        valid, _ = compute_validity(mols, log=False, position=1)
        dataset_smiles.extend(valid)
    
    # Get unique list of SMILES and save them
    dataset_smiles = list(set(dataset_smiles))
    with open(save_path, 'w') as f:
        for smiles in [s for s in dataset_smiles if s != '']:
            f.write(smiles + '\n')

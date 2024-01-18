import torch
import os
from tqdm import tqdm

from opts import parse_arguments_sample
from utils import *
from metrics import *

def sample(model, diffusion, n_samples, dataset_info, save_path:str, prefix='', position=0):
    # Sample graphs using supplied model
    X, E = diffusion.sample(model, n_samples, dataset_info, position=position)
    # Convert graphs to molecules
    mols = [mol_from_graph(X_i, E_i, dataset_info) for X_i, E_i in prepare_graph(X, E)]
    # Save the molecules
    save_molecules(mols[:20], os.path.join(save_path, f"{prefix}mol.jpg"))
    
    return mols

def main():
    mols = []
    # Parse arguments
    args = parse_arguments_sample()

    # Setup model and diffusion process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, dataset_info = get_data(args.data_name, 1)
    diffusion, model = get_model(device, dataset_info, args.n_layers)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Samples molecules in batches
    l = args.n_samples // args.batch_size + 1
    print(f"Sampling {args.n_samples} new molecules...")
    for i in tqdm(range(1, l + 1), position=0):
        samples = sample(model, diffusion, min(args.batch_size, args.n_samples - (i - 1) * args.batch_size), dataset_info, args.save_path, prefix=str(i) + '_', position=1)
        mols.extend(samples)
    
    # Report metrics: Validity, Uniqueness, and Novelty
    valid_smiles, all_smiles = compute_validity(mols)
    unique = compute_uniqueness(valid_smiles)
    novel, dataset_smiles = compute_novelty(valid_smiles, args.smiles_path)
    unovel, _ = compute_novelty(unique, args.smiles_path)

    smiles, valid, unique, novel, unovel, dset_smiles = len(all_smiles), len(valid_smiles), len(unique), len(novel), len(unovel), len(dataset_smiles)
    print(f"Samples: {smiles}, dSmiles: {dset_smiles}")
    print(f"Validity: {valid}, {100*valid/smiles:.2f}%")
    print(f"Uniqueness: {unique}, S {100*unique/smiles:.2f}%, V {100*unique/valid:.2f}%")
    print(f"Novelty: {novel}, S {100*novel/smiles:.2f}%, V {100*novel/valid:.2f}%")
    print(f"UniqueNovel: {unovel}, {100*unovel/unique:.2f}%")

if __name__ == '__main__':
    main()
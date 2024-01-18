import os
import torch
from tqdm import tqdm
from torch import optim

from utils import *
from metrics import *
from model_utils import *
from opts import *
from sample import *

def train(args):
    # Setup folder for checkpoints and training device
    setup_logging(args.save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloader, model, diffusion, optimiser, and loss
    train, val, dataset_info = get_data(args.data_name, args.batch_size)
    diffusion, model = get_model(device, dataset_info, args.n_layers)
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12, amsgrad=True)
    mse_X = torch.nn.MSELoss()
    mse_E = torch.nn.MSELoss()
    mse_y = torch.nn.MSELoss()

    # Resume run from epoch if argument is set
    if args.resume_epoch > 1:
        print(f'Resuming from epoch {args.resume_epoch}')
        model, optimiser = load_checkpoint(model, optimiser, args.save_path, args.resume_epoch)

    # Compute smiles for the training dataset to compute metrics when sampling
    compute_dataset_smiles(train, dataset_info, os.path.join(args.save_path, 'smiles.txt'))

    # Training - Iterate every epoch
    for epoch in range(args.resume_epoch, args.epochs + 1):
        # Logging and tqdm
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(train)
        # Iteratate every batch of graphs
        for graph in pbar:
            # Send node features, dense adjacency matrix and labels to device
            X, E, node_mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            X = X.to(device); E = E.to(device); y = graph.y.to(device); node_mask = node_mask.to(device)
            # Get timestep t, noise and noisy graphs
            t = diffusion.sample_timesteps(y.size(0)).to(device)
            X_t, E_t, _, noise_X, noise_E, noise_y = diffusion.forward_process(X, E, y, t, node_mask)
            # Predict noise and calculate loss
            y_t = torch.hstack((y[:,:-1], t.unsqueeze(-1)))
            pred_X, pred_E, pred_y = model(X_t, E_t, y_t, node_mask)
            X_loss = mse_X(pred_X, noise_X) if noise_X.numel() > 0 else torch.tensor(0.0, device=device)
            E_loss = mse_E(pred_E, noise_E) if noise_E.numel() > 0 else torch.tensor(0.0, device=device)
            y_loss = mse_y(pred_y, noise_y) if y.numel() > 0 else torch.tensor(0.0, device=device)
            loss = X_loss + E_loss + y_loss

            # Propogate loss: Reset gradients, calculate new gradients, take gradient descent step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update logging
            pbar.set_postfix(MSE=loss.item())

        # Sample and generate checkpoint
        if epoch % 5 == 0:
            sample(model, diffusion, 20, dataset_info, os.path.join(args.save_path, 'samples'), f'epoch_{epoch}_')
            create_checkpoint(model, optimiser, args.save_path, epoch)
            evaluate(model, diffusion, dataset_info, val, epoch, device, args)

def evaluate(model, diffusion, dataset_info, validation_set, epoch, device, args):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Get random data point from validation set
        _, data = list(enumerate(validation_set))[0]
        # Send node features, dense adjacency matrix and labels to device
        X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        X = X.to(device); E = E.to(device); y = data.y.to(device); node_mask = node_mask.to(device)
        # Get timestep t and noisy graphs
        i = torch.randint(low=1, high=diffusion.noise_steps).item()
        t = (torch.ones(size=y.size(0)) * i).long().to(device)
        X_t, E_t, *_ = diffusion.forward_process(X, E, y, t, node_mask)
        # Predict clean graph
        X_0, E_0, _ = diffusion.reverse_process(y.size(0), model, node_mask, X_t, E_t, y)

        # Convert to molecules and save
        X_0, E_0 = mask_nodes(*continuous_to_discrete(X_0, E_0), node_mask)
        X_t, E_t = mask_nodes(*continuous_to_discrete(X_t, E_t), node_mask)
        mols = [mol_from_graph(x_i, e_i, dataset_info) for x_i, e_i in prepare_graph(X, E)]
        mols_0 = [mol_from_graph(x_i, e_i, dataset_info) for x_i, e_i in prepare_graph(X_0, E_0)]
        mols_t = [mol_from_graph(x_i, e_i, dataset_info) for x_i, e_i in prepare_graph(X_t, E_t)]
        save_molecules(mols, os.path.join(args.save_path, 'eval', f'{epoch}_{i}_val_orig.jpg'))
        save_molecules(mols_t, os.path.join(args.save_path, 'eval', f'{epoch}_{i}_val_noisy.jpg'))
        save_molecules(mols_0, os.path.join(args.save_path, 'eval', f'{epoch}_{i}_val_clean.jpg'))

    model.train()

def main():
    args = parse_arguments_train()
    train(args)


if __name__ == '__main__':
    main()

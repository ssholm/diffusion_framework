import os
import torch
from tqdm import tqdm

from utils import *
from opts import *
from sample import *

def train(args):
    # Setup folder for checkpoints and training device
    setup_logging(args.save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloader, model, diffusion, optimiser, and loss
    dataloader = get_data(args.image_size, args.data_path, args.batch_size)
    diffusion, model = get_model(args, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    # Resume run from epoch if argument is set
    if args.resume_epoch > 1:
        print(f'Resuming from epoch {args.resume_epoch}')
        model, optimiser = load_checkpoint(model, optimiser, args.save_path, args.resume_epoch)

    # Training - Iterate every epoch
    for epoch in range(args.resume_epoch, args.epochs + 1):
        # Logging and tqdm
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        # Iteratate every batch of images
        for (images, labels) in pbar:
            # Send images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Get timestep t, noisy images and applied noise
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.forward_process(images, t)

            # Predict noise and calculate loss
            pred_noise = model(x_t, t, labels)
            loss = mse(noise, pred_noise)

            # Propogate loss: Reset gradients, calculate new gradients, take gradient descent step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update logging
            pbar.set_postfix(MSE=loss.item())

        # Sample and generate checkpoint
        if epoch % 5 == 0:
            sample(model, diffusion, 20, args.n_classes, os.path.join(args.save_path, 'samples'), device, f'epoch_{epoch}_')
            create_checkpoint(model, optimiser, args.save_path, f'epoch_{epoch}_')


def main():
    args = parse_arguments_train()
    train(args)


if __name__ == '__main__':
    main()

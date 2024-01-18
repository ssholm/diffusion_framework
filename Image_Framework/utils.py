import os
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

from diffusion import *
from models import *

def get_model(args, device):
    # Setup model and diffusion process on device
    diffusion = Diffusion(img_size=args.image_size, device=device)
    model = UNet(num_classes=args.n_classes, device=device)
    model = torch.nn.DataParallel(model)
    model.to(device)

    return diffusion, model

def create_checkpoint(model, optim, save_path, prefix):
    # Save model and optimiser parameters as a checkpoint
    torch.save(model.state_dict(), os.path.join(save_path, "models", f'{prefix}_model.pt'))
    torch.save(optim.state_dict(), os.path.join(save_path, "models", f'{prefix}_optim.pt'))

def load_checkpoint(model, optim, save_path, epoch):
    # Load model and optimiser parameters from a checkpoint
    model.load_state_dict(torch.load(os.path.join(save_path, "models", f'epoch_{epoch}_model.pt')))
    optim.load_state_dict(torch.load(os.path.join(save_path, "models", f'epoch_{epoch}_optim.pt')))

    return model, optim

def save_images(images, path):
    # Save generated images to path
    # Create grid of images
    grid = torchvision.utils.make_grid(images)
    # Convert grid to numpy array
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # Convert numpy array to images and save images
    img = Image.fromarray(ndarr)
    img.save(path)


def get_data(image_size: int, data_path:str, batch_size:int):
    # Loads data from folder, transforms and normalises it
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load dataset from folder and transform it
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
    # Create dataloader to iterate dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def setup_logging(save_path):
    # Initialises folder structure for training, evaluation and sampling
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "eval"), exist_ok=True)

import torch
import os

from opts import parse_arguments_sample
from utils import *

def sample(model, diffusion, n_samples, n_classes, save_path:str, device, prefix='', position=0):
    # Sample images using the supplied model
    # Create labels for images to generate
    labels = (torch.arange(n_samples) % n_classes).long().to(device)
    # Sample images
    sampled_images = diffusion.reverse_process(model, n=n_samples, labels=labels, position=position)
    # Save images
    save_images(sampled_images, os.path.join(save_path, f"{prefix}img.jpg"))

def main():
    # Parse arguments
    args = parse_arguments_sample()

    # Setup model and diffusion process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion, model = get_model(args, device)
    model.load_state_dict(torch.load(args.model_path))

    # Sample images in batches
    print(print(f"Sampling {args.n_samples} new images..."))
    factor = 20
    l = args.n_samples // factor
    for i in tqdm(l):
        sample(model, diffusion, factor, args.n_classes, args.save_path, device, prefix=str(i) + '_', position=1)

if __name__ == '__main__':
    main()
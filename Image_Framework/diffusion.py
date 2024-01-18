import torch
from tqdm import tqdm

class Diffusion:
    # A class to implement all the diffusion processes
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Precompute noise schedule variables
        self.beta, self.alpha, self.alpha_bar = self.linear_schedule()

    def linear_schedule(self):
        # https://arxiv.org/abs/2006.11239
        # Computes the linear noise schedule variables: beta, alpha, alpha_bar
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        return beta, alpha, alpha_bar

    def forward_process(self, x, t):
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * Ɛ
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        # Sample noise from standard Gaussian distribution
        noise = torch.randn_like(x)

        # Apply noise to images and return noisy images and applied noise
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise

    def sample_timesteps(self, n):
        # Sample a batch of timesteps from a uniform distribution
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def reverse_process(self, model, n, labels=None, position=0):
        # Handles the reverse process of the diffusion model
        model.eval()
        with torch.no_grad():
            # Sample n noisy images from a standard Gaussian distribution
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # Iterate every timestep in reverse
            for i in tqdm(range(self.noise_steps - 1, -1, -1), position=position, leave=(position==0)):
                # Batch of timesteps
                t = (torch.ones(n) * i).long().to(self.device)

                # Predict noise
                pred_noise = model(x, t, labels)

                # Prepare noise schedule variables for timestep t
                alpha = self.alpha[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Do not include noise in the last step
                noise = torch.zeros_like(x) if i == 0 else torch.randn_like(x)

                # x_t-1 = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * pred_noise) + sqrt(beta_t) * noise
                x = 1 / torch.sqrt(alpha) * (x - (beta / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(beta) * noise
            
        model.train()

        # Clips the output to (-1, 1), squeezes it to (0, 1) and unsqueezes it to (0, 255) to represent images
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

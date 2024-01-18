import torch
from tqdm import tqdm

from model_utils import *

class Diffusion:
    # A class to implement all the diffusion processes
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, device="cuda", schedule='cos'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Precompute noise schedule
        if schedule == 'cos':
            beta, alpha, alpha_bar = self.cosine_schedule()
        elif schedule == 'lin':
            beta, alpha, alpha_bar = self.linear_schedule()
        else:
            beta, alpha, alpha_bar = self.cosine_schedule()

        self.beta, self.alpha, self.alpha_bar = beta.to(device), alpha.to(device), alpha_bar.to(device)

    def linear_schedule(self):
        # https://arxiv.org/abs/2006.11239
        # Computes the linear noise schedule variables: beta, alpha, alpha_bar
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        return beta, alpha, alpha_bar

    def cosine_schedule(self):
        # https://openreview.net/forum?id=-NEXDKk8gZ
        # s parameter proposed in the paper
        s = 0.008
        # Initialise all t values in a list
        t = torch.arange(0, self.noise_steps + 1)

        # f(t) = cos(((t / T) + 1) / (1 + s) * 0.5 * pi)^2
        f_t = torch.cos(((t / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        # alpha_tilde = f(t) / f(0)
        alpha_tilde = f_t / f_t[0]
        # beta = 1 - (alpha_tilde_t / alpha_tilde_{t-1})
        beta = 1 - (alpha_tilde[1:] / alpha_tilde[:-1])
        # Clip betas to predefined range to avoid too large and small values
        beta = torch.clamp(beta, min=0., max=0.999)

        # Compute alpha and alpha_bar based on formulas
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        return beta, alpha, alpha_bar

    def sample_timesteps(self, n):
        # Sample a batch of timesteps from a uniform distribution
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def forward_process(self, X, E, y, t, node_mask):
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * Ɛ
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]

        # Sample noise
        noise_X = torch.randn_like(X)
        noise_E = torch.randn_like(E)
        noise_y = torch.randn_like(y)

        # Mirror edge features to ensure graph is undirected and mask unused nodes
        noise_E = mirror(noise_E)
        noise_X, noise_E = mask_nodes(noise_X, noise_E, node_mask)

        # Apply noise to graph
        X_t = sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise_X
        E_t = sqrt_alpha_bar.unsqueeze(1) * E + sqrt_one_minus_alpha_bar.unsqueeze(1) * noise_E
        y_t = sqrt_alpha_bar.squeeze(1) * y + sqrt_one_minus_alpha_bar.squeeze(1) * noise_y

        # Mask unused nodes and return noisy graphs and applied noise
        X_t, E_t = mask_nodes(X_t, E_t, node_mask)
        return X_t, E_t, y_t, noise_X, noise_E, noise_y

    def sample(self, model, n, dataset_info, position=0):
        # Sample molecules
        model.eval()
        with torch.no_grad():
            # Sample number of nodes in each molecule from uniform distribution
            nodes = torch.randint(low=dataset_info.min_nodes, high=dataset_info.max_nodes, size=(n,)).to(self.device)
            max_nodes = torch.max(nodes).item()

            # Generate node mask
            arange = torch.arange(max_nodes, device=self.device).unsqueeze(0).expand(n, -1)
            node_mask = arange < nodes.unsqueeze(1)
            node_mask = node_mask.float()

            # Sample n noisy graphs from a standard Gaussian distribution
            X = torch.randn((n, max_nodes, dataset_info.input_dims['X'])).to(self.device)
            E = torch.randn((n, max_nodes, max_nodes, dataset_info.input_dims['E'])).to(self.device)
            y = torch.randn((n, dataset_info.output_dims['y'])).to(self.device)

            # Mirror edge features to ensure graph is undirected and mask unused nodes
            E = mirror(E)
            X, E = mask_nodes(X, E, node_mask)

            # Clean the graph using the reverse process
            X, E, y = self.reverse_process(n, model, node_mask, X, E, y, position)

        model.train()

        return mask_nodes(*continuous_to_discrete(X, E), node_mask)
    
    def reverse_process(self, n, model, node_mask, X, E, y, position=0):
        # Remove noise from graph using trained model
        for i in tqdm(range(self.noise_steps - 1, -1, -1), position=position, leave=(position==0)):
            # Batch of timesteps
            t = (torch.ones(n) * i).long().to(self.device)
            y = torch.hstack((y[:, :-1], t.unsqueeze(-1)))

            # Predict noise in graph
            pred_X, pred_E, y = model(X, E, y, node_mask)

            # Prepare noise schedule variables for timestep t
            beta_X = self.beta[t][:, None, None]
            alpha_X = self.alpha[t][:, None, None]
            alpha_bar_X = self.alpha_bar[t][:, None, None]
            beta_E = self.beta[t][:, None, None, None]
            alpha_E = self.alpha[t][:, None, None, None]
            alpha_bar_E = self.alpha_bar[t][:, None, None, None]

            # Do not include noise in the last step
            noise_X = torch.zeros_like(X) if i == 0 else torch.randn_like(X)
            noise_E = torch.zeros_like(E) if i == 0 else torch.randn_like(E)
            
            # Mirror edge features to ensure graph is undirected and mask unused nodes
            noise_E = mirror(noise_E)
            noise_X, noise_E = mask_nodes(noise_X, noise_E, node_mask)

            # x_t-1 = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * pred_noise) + sqrt(beta_t) * noise
            X = 1 / torch.sqrt(alpha_X) * (X - (beta_X / (torch.sqrt(1 - alpha_bar_X))) * pred_X) + torch.sqrt(beta_X) * noise_X
            E = 1 / torch.sqrt(alpha_E) * (E - (beta_E / (torch.sqrt(1 - alpha_bar_E))) * pred_E) + torch.sqrt(beta_E) * noise_E
        
        return X, E, y
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
import torch_geometric as pyg

from model_utils import *

class PNAX(nn.Module):
    def __init__(self, dx, dy):
        # Map node features to global features
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        # X: bs, n, dx
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out
    

class PNAE(nn.Module):
    def __init__(self, de, dy):
        # Map edge features to global features.
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E):
        # E: bs, n, n, de
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Transformer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5) -> None:
        super().__init__()

        self.attn = NodeEdgeBlock(dx, de, dy, n_head)

        self.transX = pyg.nn.Sequential('X, newX', [
            (Dropout(dropout), 'newX -> newX_d'),
            (torch.add, 'X, newX_d -> X'),
            (LayerNorm(dx, eps=layer_norm_eps), 'X -> X'),
            (Linear(dx, dim_ffX), 'X -> outX'),
            (F.relu, 'outX -> outX'),
            (Dropout(dropout), 'outX -> outX'),
            (Linear(dim_ffX, dx), 'outX -> outX'),
            (Dropout(dropout), 'outX -> outX'),
            (torch.add, 'X, outX -> X'),
            (LayerNorm(dx, eps=layer_norm_eps), 'X -> X')
        ])
        
        self.transE = pyg.nn.Sequential('E, newE', [
            (Dropout(dropout), 'newE -> newE_d'),
            (torch.add, 'E, newE_d -> E'),
            (LayerNorm(de, eps=layer_norm_eps), 'E -> E'),
            (Linear(de, dim_ffE), 'E -> outE'),
            (F.relu, 'outE -> outE'),
            (Dropout(dropout), 'outE -> outE'),
            (Linear(dim_ffE, de), 'outE -> outE'),
            (Dropout(dropout), 'outE -> outE'),
            (torch.add, 'E, outE -> E'),
            (LayerNorm(de, eps=layer_norm_eps), 'E -> E')
        ])


        self.trans_y = pyg.nn.Sequential('y, new_y', [
            (Dropout(dropout), 'new_y -> new_y_d'),
            (torch.add, 'y, new_y_d -> y'),
            (LayerNorm(dy, eps=layer_norm_eps), 'y -> y'),
            (Linear(dy, dim_ffy), 'y -> outy'),
            (F.relu, 'outy -> outy'),
            (Dropout(dropout), 'outy -> outy'),
            (Linear(dim_ffy, dy), 'outy -> outy'),
            (Dropout(dropout), 'outy -> outy'),
            (torch.add, 'y, outy -> y'),
            (LayerNorm(dy, eps=layer_norm_eps), 'y -> y')
        ])

    def forward(self, X, E, y, node_mask):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        newX, newE, new_y = self.attn(X, E, y, node_mask)
        X = self.transX(X, newX)
        E = self.transE(E, newE)
        y = self.trans_y(y, new_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = PNAX(dx, dy)
        self.e_y = PNAE(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def masked_softmax(self, x, mask, **kwargs):
        # Softmax using mask before computation
        if mask.sum() == 0: return x
        x_masked = x.clone()
        x_masked[mask == 0] = -float("inf")
        return torch.softmax(x_masked, **kwargs)

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / torch.sqrt(torch.tensor(Y.size(-1))).item()

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = self.masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, dataset_info):
        super().__init__()
        self.n_layers = n_layers
        input_dims = dataset_info.input_dims
        output_dims = dataset_info.output_dims
        hidden_dims = dataset_info.hidden_dims
        hidden_mlp_dims = dataset_info.hidden_mlp_dims

        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), nn.ReLU())

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), nn.ReLU())

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), nn.ReLU())

        self.tf_layers = nn.ModuleList([Transformer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        X = self.mlp_in_X(X)
        X, E = mask_nodes(X, new_E, node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        X, E = mask_nodes(X, E, node_mask)
        return X, E, y
    

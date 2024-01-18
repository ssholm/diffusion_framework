import torch
import torch_geometric as pyg

def continuous_to_discrete(X, E):
    # Convert continouos values to discrete to represent graphs
    # Convert node features
    _, idx = torch.max(X, dim=2, keepdim=True)
    X = torch.zeros_like(X, dtype=torch.int32)
    X.scatter_(2, idx, 1.).type(dtype=torch.int32)

    # Convert edge fewatures
    _, idx = torch.max(E, dim=3, keepdim=True)
    E = torch.zeros_like(E, dtype=torch.int32)
    E.scatter_(3, idx, 1.).type(dtype=torch.int32)

    return X, E

def to_dense(X, edge_index, edge_attr, batch):
    # Convert node features to batch representation
    X, node_mask = pyg.utils.to_dense_batch(x=X, batch=batch)
    # Remove self loops
    edge_index, edge_attr = pyg.utils.remove_self_loops(edge_index, edge_attr)
    # Create dense adjacency matrix with edge features
    max_nodes = X.size(1)
    E = pyg.utils.to_dense_adj(edge_index=edge_index, edge_attr=edge_attr, batch=batch, max_num_nodes=max_nodes)

    # Encode no edge attribute
    E = encode_no_edge(E)

    # Mask unused nodes and return graph and node mask
    X, E = mask_nodes(X, E, node_mask)
    return X, E, node_mask

def encode_no_edge(E):
    # Get mask indicating no edge
    no_edge = torch.sum(E, dim=3) == 0
    # Set no edge attribute
    E[:, :, :, 0][no_edge] = 1
    # Remove diagonal
    diag = torch.eye(E.size(1), dtype=torch.bool).unsqueeze(0).expand(E.size(0), -1, -1)
    E[diag] = 0

    return E

def mask_nodes(X, E, node_mask):
    # Create masks for nodes and edges
    X_mask = node_mask.unsqueeze(-1)          # bs, n, 1
    E_mask1 = X_mask.unsqueeze(2)             # bs, n, 1, 1
    E_mask2 = X_mask.unsqueeze(1)             # bs, 1, n, 1
    
    # Apply masks
    X = X * X_mask
    E = E * E_mask1 * E_mask2
    
    return X, E

def mirror(E):
    # Use only upper triangular part for undirected graphs and mirror to lower triangular part
    # Get inidices for upper triangular part
    upper_indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)
    # Create mask for upper triangular part
    upper_mask = torch.zeros_like(E)
    upper_mask[:, upper_indices[0], upper_indices[1], :] = 1
    # Apply mask and copy to lower triangular part
    E = E * upper_mask
    E = E + torch.transpose(E, 1, 2)
    
    return E


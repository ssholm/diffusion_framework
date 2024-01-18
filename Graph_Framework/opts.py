import argparse

def parse_arguments_train():
    # Parse command lind arguments for training
    # Create parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-e', '--epochs', action='store', default=1000, type=int, required=False, help='Number of epochs to run', metavar='epochs', dest='epochs')
    parser.add_argument('-bs', '--batch-size', action='store', default=512, type=int, required=False, help='The batch size', metavar='batch size', dest='batch_size')
    parser.add_argument('-dn', '--data-name', action='store', type=str, required=True, help='Name of dataset', metavar='data path', dest='data_name')
    parser.add_argument('-sp', '--save-path', action='store', type=str, required=True, help='Path to save folder', metavar='save path', dest='save_path')
    parser.add_argument('-lr', '--lr', action='store', default=3e-4, type=float, required=False, help='Learning rate', metavar='learning rate', dest='lr')
    parser.add_argument('-nl', '--n-layers', action='store', default=6, type=int, required=False, help='Number of transformer layers', metavar='nl', dest='n_layers')
    parser.add_argument('-re', '--resume-epoch', action='store', default=1, type=int, required=False, help='Epoch to resume run from', metavar='resume from epoch', dest='resume_epoch')

    # Parse and return arguments
    args = parser.parse_args()
    return args

def parse_arguments_sample():
    # Parse command lind arguments for sampling
    # Create parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-mp', '--model-path', action='store', type=str, required=True, help='Path to model file', metavar='model-path', dest='model_path')
    parser.add_argument('-sp', '--save-path', action='store', type=str, required=True, help='Path to save folder', metavar='save path', dest='save_path')
    parser.add_argument('-smp', '--smiles-path', action='store', type=str, required=True, help='Path to smiles file', metavar='smiles path', dest='smiles_path')
    parser.add_argument('-s', '--samples', action='store', default=10000, type=int, required=False, help='Number of samples', metavar='s', dest='n_samples')
    parser.add_argument('-bs', '--batch-size', action='store', default=20, type=int, required=False, help='The batch size for samples', metavar='batch size', dest='batch_size')
    parser.add_argument('-nl', '--n-layers', action='store', default=6, type=int, required=False, help='Number of transformer layers', metavar='nr', dest='n_layers')
    parser.add_argument('-dn', '--data-name', action='store', type=str, required=True, help='Name of dataset', metavar='data name', dest='data_name')

    # Parse and return arguments
    args = parser.parse_args()
    return args

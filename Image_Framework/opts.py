import argparse

def parse_arguments_train():
    # Parse command lind arguments for training
    # Create parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-e', '--epochs', action='store', default=500, type=int, required=False, help='Number of epochs to run', metavar='#epochs', dest='epochs')
    parser.add_argument('-bs', '--batch-size', action='store', default=12, type=int, required=False, help='The batch size', metavar='batch size', dest='batch_size')
    parser.add_argument('-is', '--image-size', action='store', default=64, type=int, required=False, help='Image size in the data set', metavar='image size', dest='image_size')
    parser.add_argument('-dp', '--data-path', action='store', type=str, required=True, help='Path to data folder', metavar='data path', dest='data_path')
    parser.add_argument('-sp', '--save-path', action='store', type=str, required=True, help='Path to save folder', metavar='save path', dest='save_path')
    parser.add_argument('-lr', '--lr', action='store', default=3e-4, type=float, required=False, help='Learning rate', metavar='lr', dest='lr')
    parser.add_argument('-n', '--classes', action='store', default=1, type=int, required=False, help='Number of classes', metavar='#classes', dest='n_classes')
    parser.add_argument('-re', '--resume-epoch', action='store', default=1, type=int, required=False, help='Epoch to resume run from', metavar='re', dest='resume_epoch')

    # Parse and return arguments
    args = parser.parse_args()
    return args

def parse_arguments_sample():
    # Parse command lind arguments for sampling
    # Create parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-mp', '--model-path', action='store', type=str, required=True, help='Path to model file', metavar='model-path', dest='model_path')
    parser.add_argument('-sp', '--save-path', action='store', type=str, required=True, help='Path to save folder', metavar='save', dest='save_path')
    parser.add_argument('-is', '--image-size', action='store', default=64, type=int, required=True, help='Image size in the data set trained on', metavar='image', dest='image_size')
    parser.add_argument('-n', '--classes', action='store', default=1, type=int, required=False, help='Number of classes', metavar='n', dest='n_classes')
    parser.add_argument('-s', '--samples', action='store', default=100, type=int, required=False, help='Number of samples', metavar='s', dest='n_samples')

    # Parse and return arguments
    args = parser.parse_args()
    return args

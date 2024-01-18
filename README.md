# GraphGens

Code for a prespecialisation project at Aalborg University on 3rd semester of a Masters in Computer Science.

## Image_Framework
A simple implementation of the image diffusion process proposed in https://arxiv.org/abs/2006.11239.

### Training using the framework
To train a model using the framework, make sure the model is implemented in `models.py` and added to the possible list of models in the `get_model` function in `utils.py`.

Training is then performed with:\
`python3 main.py -dp data/images -sp runs/images`\
Where your training data is placed in `data/images` and checkpoints are saved to `runs/images`.

For a list of available arguments for running the training process use:\
`python3 main.py -h`

### Sampling using the framework
To sample using the framework and a pretrained model use:\
`python3 sample.py -mp runs/images/model1.pt -sp runs/images/samples -is 32`\
Where your model is saved to `runs/images/model1.pt`, your samples will be saved to `runs/images/sample` and the image size used in the training dataset is `32`.

For a list of available arguments for running the sampling process use:\
`python3 sample.py -h`

## Graph_Framework
A simple diffusion framework for working with graph data to generate new molecules in de novo drug design.\
The implementation converts the process implemented in `Image_Framework` into considering graph data instead.

### Training using the framework
To train a model using the framework, make sure the model is implemented in `models.py` and added to the possible list of models in the `get_model` function in `utils.py`.
Additionally, the desired dataset should be implemented in `datasets.py` and added to the possible list of datasets in the `get_data` function in `utils.py`.

Training is then performed with:\
`python3 main.py -dn qm9 -sp runs/qm9`\
Where your training data is the QM9 dataset and checkpoints and samples are saved to `runs/qm9`.

For a list of available arguments for running the training process use:\
`python3 main.py -h`

### Sampling using the framework
To sample using the framework and a pretrained model use:\
`python3 sample.py -mp runs/qm9/models/model1.pt -sp runs/qm9/test -dn qm9 -smp runs/qm9/smiles.txt`\
Where your model is saved to `runs/qm9/models/model1.pt`, your samples will be saved to `runs/qm9/test`, the dataset used for training is `qm9` and the SMILES of the training dataset is saved to `runs/qm9/smiles.txt`.

For a list of available arguments for running the sampling process use:\
`python3 sample.py -h`


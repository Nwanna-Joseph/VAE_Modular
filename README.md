#  Implementation of a Variational Autoencoder using Pytorch :white_check_mark:

## Contributors (Group Members)
 1) 🇨🇩: Junior Kaningini (https://www.linkedin.com/in/junior-kaningini-a02442196/) (nkaningini@uaimsammi.org)
 2) :kenya: Angela Munayo (https://www.linkedin.com/in/angela-munyao-110777119/) (amusangi@aimsammi.org)
 3) :rwanda: Gasana Elysee (https://www.linkedin.com/in/elysee-manimpire-gasana) (emanimpire@aimsammi.org)
 4) :nigeria: Nwanna Joseph (https://ng.linkedin.com/in/nnaemeka-nwanna-090b9b142) (jnwanna@aimsammi.org)

## Description:

The variational autoencoder (VAE) is a framework for training two neural networks: an encoder and a decoder, in order to learn a mapping from a high-dimensional data representation to a lower-dimensional space and back again

## Presentation:
https://github.com/Junior-081/VAE_Modular/blob/patch-3/Variational_Auto_Encoder__VAEs_.pdf

## Commands:
 1) To train the network ```python train.py```
 2) To generate new SMILE ```python generate.py```
 3) To install dependencies ```pip install -r requirements.txt ```

## Dependencies:
 1) numpy
 2) torch
 3) argparse
 4) torchvision

## :card_file_box: Folder Structure:
 1) data : Contains files related to the SMILES data
 2) tests: Contains scripts for unit testing 
 3) saved_model: Where models that have been trained are saved

## :card_index_dividers: Files : 
 1) data/SMILEDataset : The SMILE dataset class built on pytorch
 2) data/smiles.txt: The training data
 3) args_params.py : Manages the extraction of the command line arguments
 4) encode_data.py : Manages one-hot encoding and decoding
 5) main.py: Manages the training of the network
 6) model.py : Manages the Variational Autoencoder Neural Network class for encoding, decoding and reparameterization
 7) utils.py : Manages the utility functions for softmax, temperature sampling, one-hot encoding and decoding, loss function, generating new models, e.t.c

## Special Thanks to :
 1) The organizers of the AMMI 2022 program

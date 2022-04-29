import numpy as np
import torch
from torch.utils.data import DataLoader
from data.SMILEDataset import SMILEDataset
from model import VAE
from utils import flatten, sequence_to_idx, loss_fn
from args_params import args

if __name__ == '__main__':

    batch_size = args.batch_size
    lr = args.lr

    trainloader = DataLoader(SMILEDataset(), batch_size=32)
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mu_learnt = None
    std_learnt = None
    loss_array = []
    bce_loss_array = []
    kld_loss_array = []
    for epoch in range(2):
        for idx, image in enumerate(trainloader):
            # images = [print(sequence_to_idx( a ).shape) for a in image ]
            images = flatten(
                torch.tensor(np.array([sequence_to_idx(a) for a in image]), dtype=torch.float32).reshape(-1, 1, 1, 35))

            recon_images, mu, logvar = model(images)
            mu_learnt = mu
            std = logvar.mul(0.5).exp_()
            std_learnt = std

            loss, bce_loss, kld_loss = loss_fn(recon_images, images, mu, logvar)
            loss_array.append(loss)
            bce_loss_array.append(bce_loss)
            kld_loss_array.append(kld_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, 2, loss.item() / 32))

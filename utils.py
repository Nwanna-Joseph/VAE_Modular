import numpy as np
import torch
import torchvision as torchvision
from torch.autograd import Variable
from torch.nn.functional import mse_loss

from encode_data import SMILESEncoder

smiles_encoder = SMILESEncoder()


def softmax(z):
    """Softmax function """
    # raise NotImplementedError
    return torch.nn.functional.softmax(z, dim=0)


def idx_to_sequence(idx):
    return smiles_encoder.decode(idx)


def sequence_to_idx(sequence):
    return smiles_encoder.encode(sequence)


def temperature_sampling(temperature):
    """
    Temperature sampling wrapper function
    This wrapper function will allow us use the temperature sampling strategy to decode our predicted sequences
    """

    def decode(preds):
        """
        Decoder using temperature
        """

        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        reweighted_preds = softmax(torch.tensor(preds))
        probs = np.random.multinomial(3, reweighted_preds, 3)

        return np.argmax(probs)

    return decode


def loss_fn(recon_x, x, mu, logvar):
    BCE = mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE + KLD, BCE, KLD


def flatten(x):
    return to_var(x.view(x.size(0), -1))


def save_image(x, path='real_image.png'):
    torchvision.utils.save_image(x, path)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def generate(model, mu, std):
    model.eval()
    z = np.random.normal(mu.detach().numpy(),
                         std.detach().numpy())
    out_probs = model.decoder(torch.tensor(z, dtype=torch.float32))
    _images_ = out_probs.split(100, dim=0)

    output = ""
    for row in _images_[0].detach().numpy():
        dx = temperature_sampling(1)(row)
        ch = smiles_encoder.i2c.get(dx)
        if ch:
            output += ch
    generation = output
    return generation

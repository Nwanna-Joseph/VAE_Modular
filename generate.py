import torch

from utils import generate

if __name__ == '__main__':
    mu_learnt, std_learnt, model = torch.load("./saved_model/mu_tensor.pt")
    text = generate(model, mu_learnt, std_learnt)
    print(text)

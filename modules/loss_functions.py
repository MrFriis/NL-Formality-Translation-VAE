import torch
from torch.nn import functional as F


def ELBO_loss(recon_sentence, sentence, mu, log_var):
    CE = F.cross_entropy(recon_sentence, sentence, reduction='none')
    CE = CE.sum(1).mean()  # sum over input mean over batch
    KLD = -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=2)  # sum over latent dim
    ELBO = CE + KLD.mean()  # mean over batch

    return ELBO, CE, KLD.sum()


def ELBO_lAnneal(recon_sentence, sentence, mu, log_var, epoch, maxEpoch=10):
    CE = F.cross_entropy(recon_sentence, sentence, reduction='none')
    CE = CE.sum(1).mean()  # sum over input mean over batch
    KLD = -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=2)  # sum over latent dim

    if epoch == 0:
        anneal = 0
    elif epoch < maxEpoch:
        anneal = maxEpoch / epoch  # This here wonøt work for more than 11
    else:
        anneal = 1

    ELBO = CE + KLD.mean() * anneal

    return ELBO, CE, KLD.sum()

def ELBO_cAnneal(recon_sentence, sentence, mu, log_var, epoch, cyclic=100):
    """
    with cyclical annealing as described in
    https://github.com/haofuml/cyclical_annealing

    ______________________________________________________
    mu has size [1,batchSize,latentDim]

    """
    CE = F.cross_entropy(recon_sentence, sentence, reduction='none')
    CE = CE.sum(1).mean()  # sum over input mean over batch
    KLD = - 0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=2)  # sum over latent dim

    anneal = cycAnneal(epoch, cyclic)

    ELBO = CE + KLD.mean() * anneal
    return ELBO, CE, KLD.sum()


def ELBO_cAnnealFB(recon_sentence, sentence, mu, log_var, epoch, lamb=3, cyclic=20):
    """
    This is the elementwise freebits described as (FB) in the paper
    https://arxiv.org/pdf/1909.00868.pdf

    with cyclical annealing as described in
    https://github.com/haofuml/cyclical_annealing

    """
    CE = F.cross_entropy(recon_sentence, sentence, reduction='none')
    CE = CE.sum(1).mean()  # sum over input mean over batch

    elementKLD = -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var))  # preserves (1,batchSize,latentDim)

    # take max(KLD, lamb) elementwise. Sum over latent dimension
    KLDFB = torch.max(elementKLD, torch.ones_like(elementKLD) * lamb).sum(2)

    anneal = cycAnneal(epoch, cyclic)

    ELBO = CE + KLDFB.mean() * anneal

    return ELBO, CE, KLDFB.sum()


def cycAnneal(epoch, cyclic):
    """
    returns anneal weight to multiply KLD with in elbo
    takes half a cycle to get to 1. for the rest of the cycle it will remain 1
    so it resembles https://github.com/haofuml/cyclical_annealing

    Function assumes epoch starts at 0
    """

    cycle = (epoch) % cyclic
    anneal = min(1, 2 / cyclic * cycle)

    return anneal

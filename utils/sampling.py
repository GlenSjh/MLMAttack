import torch

def dirichlet_softmax(logits, pi, temperature=1, hard=False, dim=-1):
    argmin = pi.sample(logits.shape[:-1]).log().to(logits.device) - logits
    y_soft = (-argmin / temperature).softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def sample_gumbel(logits, eps=1e-20):
    shape = logits.shape
    U = torch.rand(shape)
    U = U.to(logits.device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits)
    return (y / temperature).softmax(dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if (not hard) or (logits.nelement() == 0):
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


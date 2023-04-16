import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from .metrics import accuracy
from util import SmoothCrossEntropyLoss
from util import track_bn_stats

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')


def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


def adp_loss(x,y,models,criterion_ce,lamda,log_det_lamda,num_classes):
    y_true = torch.zeros(x.size(0), num_classes).cuda()
    y_true.scatter_(1, y.view(-1, 1), 1)

    loss_std = 0
    mask_non_y_pred = []
    ensemble_probs = 0

    for model in models:
        outputs = model(x)
        loss_std += criterion_ce(outputs, y)
        y_pred = F.softmax(outputs, dim=-1)
        bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))  
        mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, num_classes - 1))
        ensemble_probs += y_pred
    
    ensemble_probs = ensemble_probs / len(models)
    ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + 1e-20)),dim=-1).mean()

    mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
    
    mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1,keepdim=True)
    matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1))
    log_det = torch.logdet(matrix + 1e-6 * torch.eye(len(models), device=matrix.device).unsqueeze(0)).mean()
    loss = loss_std - lamda * ensemble_entropy - log_det_lamda * log_det

    return loss


def trades_loss(models, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, 
                attack='linf-pgd', beta=1.0, lamda=2.0, log_det_lamda=0.5, label_smoothing=0.1,num_classes=10):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    for model in models:
        model.train()
        track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    ensemble = Ensemble(models)

    p_natural = F.softmax(ensemble(x_natural), dim=1)
    p_natural = p_natural.detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(ensemble(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')

    for model in models:
        model.train()
        track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss

    loss_std = adp_loss(x_natural, y, models, criterion_ce, lamda, log_det_lamda, num_classes)

    logits_natural = ensemble(x_natural)
    logits_adv = ensemble(x_adv)
    
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))

    loss = loss_std + beta * loss_robust

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                    'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics
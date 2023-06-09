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

    return loss, torch.log(ensemble_probs)


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

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        p_natural = F.softmax(ensemble(x_natural), dim=1).detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad() and torch.autocast(device_type='cuda', dtype=torch.float16):
                out = ensemble(x_adv)
                loss_kl = criterion_kl(F.log_softmax(out, dim=1), p_natural)
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
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss_std, logits_natural = adp_loss(x_natural, y, models, criterion_ce, lamda, log_det_lamda, num_classes)
        loss_adv, logits_adv     = adp_loss(x_adv    , y, models, criterion_ce, lamda, log_det_lamda, num_classes)
        loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        loss = loss_std + loss_adv + beta * loss_robust

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                    'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics
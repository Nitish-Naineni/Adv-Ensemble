import torch
import copy
from contextlib import ExitStack
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import seed, set_bn_momentum
from models import create_model
from attacks import create_attack
from .trades import trades_loss
from attacks import CWLoss
from .metrics import accuracy

from .context import ctx_noparamgrad_and_eval
from .utils import set_bn_momentum
from .utils import seed

from .trades import trades_loss
from .cutmix import cutmix

# from torch.cuda.amp import autocast, GradScaler

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Trainer(object):
    def __init__(self,info,args):
        super(Trainer,self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed(args.seed)
        self.models = []
        self.wa_models = []
        for i in range(args.num_models):
            model = create_model(args.model, args.normalize, info)
            wa_model = copy.deepcopy(model)
            self.models.append(model)
            self.wa_models.append(wa_model)
        self.params = args
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.params.num_adv_epochs)

        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))

        self.eval_attack = create_attack(
            model=Ensemble(self.wa_models), 
            criterion=CWLoss, 
            attack_type=self.params.attack, 
            attack_eps=self.params.attack_eps, 
            attack_iter=4*self.params.attack_iter, 
            attack_step=self.params.attack_step
        )
        self.num_classes = info["num_classes"]
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data == 'cifar10':
            self.num_classes = 10
        elif self.params.data == 'cifar100':
            self.num_classes = 100
        else:
            raise ValueError(f'Invalid model name {self.params.data}!')
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps

        # self.scaler = GradScaler()
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(models):
            group_decay = []
            group_no_decay = []
            for model in models:
                for n, p in model.named_parameters():
                    if 'batchnorm' in n:
                        group_no_decay.append(p)
                    else:
                        group_decay.append(p)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups
        
        self.optimizer = torch.optim.SGD(group_weight(self.models), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        for model in self.models:
            model.train()
        
        update_iter = 0
        for data in tqdm(dataloader, desc=f'Epoch {epoch}: ', disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            for model in self.models:
                if global_step == 0:
                    # make BN running mean and variance init same as Haiku
                    set_bn_momentum(model, momentum=1.0)
                elif global_step == 1:
                    set_bn_momentum(model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            x = x.to(memory_format=torch.channels_last)

            # with autocast():
            if self.params.consistency:
                x_aug1, x_aug2, y = x[0].to(device), x[1].to(device), y.to(device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(x_aug1, x_aug2, y, beta=self.params.beta)

            else:
                if self.params.CutMix:
                    x_all, y_all = torch.empty(0), torch.empty(0)
                    for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                        x_all = torch.cat((x_all, x_tmp), dim=0)
                        y_all = torch.cat((y_all, y_tmp), dim=0)
                    x, y = x_all.to(device), y_all.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                
                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.beta is not None:
                        loss, batch_metrics = self.trades_loss(
                            x, 
                            y, 
                            beta=self.params.beta,
                            lamda=self.params.lamda,
                            log_det_lamda=self.params.log_det_lamda
                        )
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)
                    
            loss.backward()
            if self.params.clip_grad:
                for model in self.models:
                    nn.utils.clip_grad_norm_(model.parameters(), self.params.clip_grad)
            self.optimizer.step()

            # self.scaler.scale(loss).backward()
            # if self.params.clip_grad:
            #     self.scaler.unscale_(self.optimizer)
            #     for model in self.models:
            #         nn.utils.clip_grad_norm_(model.parameters(), self.params.clip_grad)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_models, self.models, global_step, decay_rate=self.params.tau, 
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        update_bn(self.wa_models, self.models) 
        return dict(metrics.mean())
    
    
    def trades_loss(self, x, y, beta, lamda, log_det_lamda):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.models, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          lamda=lamda,log_det_lamda=log_det_lamda,num_classes=self.num_classes)
        return loss, batch_metrics
    
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        for wa_model in self.wa_models:
            wa_model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = 0
            if adversarial:
                with ExitStack() as es:
                    for wa_model in self.wa_models:
                        es.enter_context(ctx_noparamgrad_and_eval(wa_model))
                    x_adv, _ = self.eval_attack.perturb(x, y)

                for wa_model in self.wa_models:
                    out += F.softmax(wa_model(x_adv), dim=-1)        
            else:
                for wa_model in self.wa_models:
                    out += F.softmax(wa_model(x), dim=-1)

            out = out / len(self.wa_models)
            out = torch.clamp(out, min=1e-40)
            out = torch.log(out)

            acc += accuracy(y, out)

        acc /= len(dataloader)
        return acc

    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': [wa_model.state_dict() for wa_model in self.wa_models], 
            'unaveraged_model_state_dict': [model.state_dict() for model in self.models]
        }, path)


    def save_model_resume(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'model_state_dict': [wa_model.state_dict() for wa_model in self.wa_models],
            'unaveraged_model_state_dict': [model.state_dict() for model in self.models],
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'epoch': epoch
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        for i, wa_model in enumerate(self.wa_models):
            wa_model.load_state_dict(checkpoint['model_state_dict'][i])

    
    def load_model_resume(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        for i, wa_model in enumerate(self.wa_models):
            wa_model.load_state_dict(checkpoint['model_state_dict'][i])
        for i, model in enumerate(self.models):
            model.load_state_dict(checkpoint['unaveraged_model_state_dict'][i])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']


def ema_update(wa_models, models, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    for model, wa_model in zip(models, wa_models):
        for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
            p_swa.data *= decay
            p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_models, models):
    """
    Update batch normalization layers.
    """
    for avg_model, model in zip(avg_models, models):
        avg_model.eval()
        model.eval()
        for module1, module2 in zip(avg_model.modules(), model.modules()):
            if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
                module1.running_mean = module2.running_mean
                module1.running_var = module2.running_var
                module1.num_batches_tracked = module2.num_batches_tracked




        
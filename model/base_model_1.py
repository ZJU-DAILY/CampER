import time
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
import torch.utils.data as Data
from sklearn import preprocessing
from tensorboardX import SummaryWriter
from apex import amp
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from utils import *


class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss


class BertEncoder(nn.Module):
    def __init__(self, args, device='cuda', alpha_aug=0.8):
        super().__init__()
        self.lm = args.lm
        self.bert = AutoModel.from_pretrained('./huggingface/roberta-base')
        self.device = device
        self.args = args
        self.alpha_aug = alpha_aug
        # linear layer
        # hidden_size = self.bert.config.hidden_size  #768
        # self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.criterion = NCESoftmaxLoss(self.device)

    def contrastive_loss(self, pos_1, pos_2, neg_value, neg_aug_value=None, hard_neg=False):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)

        if hard_neg =='true':
            l_neg = torch.bmm(pos_1.view(bsz, 1, -1), (neg_value.view(bsz, -1, 768)).transpose(1,2))  # (batch_size * topk)
            l_neg = l_neg.view(bsz,-1)
        else:
            l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        if neg_aug_value is not None:
            # l_aug_neg = torch.bmm(pos_1.view(bsz, 1, -1), neg_aug_value.view(bsz, -1, 1))
            neg_aug_value = neg_aug_value.view(bsz, -1, 768)  #(batch_size, 8, 768)

            l_aug_neg = torch.bmm(pos_1.view(bsz, 1, -1), torch.transpose(neg_aug_value, 1, 2))  #(batch_size, 768, 8)
            l_aug_neg = l_aug_neg.view(bsz, 1)
            # l_aug_neg = l_aug_neg.view(bsz, 8)
            logits = torch.cat((l_pos, l_aug_neg, l_neg), dim=1)
        else:
            logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)


    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, x1):
        x1 = x1.to(self.device)
        hidden_states = self.bert(x1)[0]  #(batch_size*256*768)
        m = nn.AdaptiveAvgPool2d((1, 768))
        out = m(hidden_states)
        out = torch.squeeze(out) #(batch_size*768)
        return out


def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


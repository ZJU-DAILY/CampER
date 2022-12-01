from os.path import abspath, dirname, join, exists
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer

device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def clip_gradients(net, dp_mechanism, dp_clip):
    if dp_mechanism == 'Laplace':
        # Laplace use 1 norm
        for k, v in net.named_parameters():
            if torch.is_tensor(v.grad):
                v.grad /= max(1, v.grad.norm(1) / dp_clip)

    elif dp_mechanism == 'Gaussian':
        # Gaussian use 2 norm
        for k, v in net.named_parameters():
            if torch.is_tensor(v.grad):
                v.grad /= max(1, v.grad.norm(2) / dp_clip)

def cal_sensitivity(lr, clip, batch_size):
    return 2 * lr * clip


def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    # print('noise_scale',noise_scale )
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    # print('noise_scale',noise_scale )
    return np.random.normal(0, noise_scale, size=size)

def add_noise(net, dp_mechanism, lr, dp_clip, dp_epsilon, batch_size, dp_delta=None):
    sensitivity = cal_sensitivity(lr, dp_clip, batch_size)
    if dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
    elif dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=dp_epsilon, delta=dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise

def save_model(model, path):
    torch.save(model.state_dict(),
               (path + "trained.pth"))


def id2tokens(path, max_len= 256):
    id2tk = dict()
    tokenizer = AutoTokenizer.from_pretrained('./huggingface/bert-base-uncased')
    lines = open(path, 'r')
    for line in lines:
        t, t_id = line.strip().split('\t')
        x = tokenizer.encode(t, max_length=max_len,truncation=True)
        x = x + [0] * (max_len - len(x))
        # x= torch.LongTensor(x)
        id2tk[int(t_id)] = x
    return id2tk

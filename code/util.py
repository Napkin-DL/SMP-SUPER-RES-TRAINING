
import codecs
import json
import logging
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models

from collections import OrderedDict

try:
    import smdistributed.modelparallel.torch as smp

except ImportError:
    pass
# import sagemaker_containers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def torch_model(model_name,
                num_classes=0,
                pretrained=True,
                local_rank=0,
                model_parallel=False):
    #     model_names = sorted(name for name in models.__dict__
    #                          if name.islower() and not name.startswith("__")
    #                          and callable(models.__dict__[name]))

    if (model_name == "inception_v3"):
        raise RuntimeError(
            "Currently, inception_v3 is not supported by this example.")

    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(model_name))
        if model_parallel:
            if local_rank == 0:
                model = models.__dict__[model_name](pretrained=pretrained)
            smp.barrier()
        model = models.__dict__[model_name](pretrained=pretrained)
    else:
        print("=> creating model '{}'".format(model_name))
        model = models.__dict__[model_name]()
    
    if model_parallel:
        model.features[1] = nn.ReLU(inplace=False)
        model.features[4] = nn.ReLU(inplace=False)
        model.features[7] = nn.ReLU(inplace=False)
        model.features[9] = nn.ReLU(inplace=False)

        model.features[12] = nn.ReLU(inplace=False)
        model.features[14] = nn.ReLU(inplace=False)
        model.features[17] = nn.ReLU(inplace=False)
        model.features[19] = nn.ReLU(inplace=False)
        
        model.classifier[1] = nn.ReLU(inplace=False)
        model.classifier[4] = nn.ReLU(inplace=False)

    if num_classes > 0:
        in_features = model.classifier[-1].out_features
        model.classifier.add_module('fc_output', nn.Linear(in_features=in_features, out_features=num_classes))
    return model


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
#         print(f"output : {output.shape}")
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    elif hasattr(t, 'index'):
        return t[0]
    else:
        return t

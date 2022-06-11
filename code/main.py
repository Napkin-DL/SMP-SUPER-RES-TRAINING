# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import print_function

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Network definition
# from model_def import Net

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import importlib

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchnet.dataset import SplitDataset
from torch.cuda.amp import autocast

class CUDANotFoundException(Exception):
    pass

import util

# from fp16 import FP16_Module, FP16_Optimizer, load_fp16_optimizer, save_fp16_optimizer

try:
    import smdistributed.modelparallel.torch as smp

except ImportError:
    pass


def train_step(args, model, data, target):
    output = model(data)
    loss = args.criterion(output, target)
    loss.backward()
    
    return output, loss


def test_step(args, model, data, target):
    output = model(data)
    test_loss = args.criterion(output, target)  # sum up batch loss
    return output, test_loss


@smp.step
def smp_train_step(args, model, data, target, scaler):
    with autocast(1 > 0):
        output = model(data)
    loss = args.criterion(output, target)
    model.backward(loss)
    
    return output, loss

@smp.step
def smp_test_step(args, model, data, target):
    output = model(data)
    test_loss = args.criterion(output, target)  # sum up batch loss
    return output, test_loss

def train(args, model, device, train_loader, optimizer, epoch, scaler):
    ##
    batch_time = util.AverageMeter('Time', ':6.3f')
    data_time = util.AverageMeter('Data', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    progress = util.ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    ##
    
    model.train()
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        if args.smp:
            output, loss = smp_train_step(args, model, data, target, scaler)
            
            if smp.tp_size() > 1:
                logits = torch.cat(tuple(output.outputs), dim=1)
            else:
                logits = torch.cat(tuple(output.outputs), dim=0)
        else:
            logits, loss = train_step(args, model, data, target)
        
        torch.cuda.empty_cache()
        
        optimizer.step()
        if args.rank == 0:
            # Measure accuracy
            prec1, prec5 = util.accuracy(logits, target, topk=(1, 5))
            
            if args.smp:
                loss = loss.reduce_mean()
            
            # to_python_float incurs a host<->device sync
            losses.update(util.to_python_float(loss.item()), data.size(0))
            top1.update(util.to_python_float(prec1), data.size(0))
            top5.update(util.to_python_float(prec5), data.size(0))

            # Waiting until finishing operations on GPU (Pytorch default: async)
            torch.cuda.synchronize()

            if batch_idx % args.log_interval == 0:
                
                batch_time.update((time.time() - start_time) / args.log_interval)
                print('Epoch: [{0}][{1}/{2}] '
                      'Train_Time={batch_time.val:.3f}: avg={batch_time.avg:.3f}, '
                      'Train_Speed={3:.3f} ({4:.3f}), '
                      'Train_Loss={loss.val:.10f}:({loss.avg:.4f}), '
                      'Train_Prec@1={top1.val:.3f}:({top1.avg:.3f}), '
                      'Train_Prec@5={top5.val:.3f}:({top5.avg:.3f})'.format(
                          epoch,
                          batch_idx,
                          len(train_loader),
                          args.world_size * args.batch_size / batch_time.val,
                          args.world_size * args.batch_size / batch_time.avg,
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))
                start_time = time.time()
                


def test(args, model, device, test_loader):
    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    progress = util.ProgressMeter(len(test_loader),
                                  [batch_time, losses, top1, top5],
                                  prefix='Test: ')
    
    model.eval()
    end = time.time()
    
    test_loss = 0
    correct = 0
    
    
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            if args.smp:
                output, loss = smp_test_step(args, model, data, target)
                
                if smp.tp_size() > 1:
                    logits = torch.cat(tuple(output.outputs), dim=1)
                else:
                    logits = torch.cat(tuple(output.outputs), dim=0)  
            else:
                logits, loss = test_step(args, model, data, target)
                
            m = {'output' : output.outputs, 'target' : target}
            torch.save(m, args.save_model + "/test_output")
            
            
            prec1, prec5 = util.accuracy(logits, target, topk=(1, 5))
            
#             test_loss += loss
#             torch.distributed.all_reduce(test_loss, group=smp.get_dp_process_group())
#             test_loss /= smp.dp_size()
#             test_loss /= batch_idx
#             test_loss = test_loss.item()
#             ppl = math.exp(loss)
            
            losses.update(util.to_python_float(test_loss), data.size(0))
            top1.update(util.to_python_float(prec1), data.size(0))
            top5.update(util.to_python_float(prec5), data.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0:
                print('Test: [{0}/{1}]  '
                      'Test_Time={batch_time.val:.3f}:({batch_time.avg:.3f}), '
                      'Test_Speed={2:.3f}:({3:.3f}), '
                      'Test_Loss={loss.val:.4f}:({loss.avg:.4f}), '
                      'Test_Prec@1={top1.val:.3f}:({top1.avg:.3f}), '
                      'Test_Prec@5={top5.val:.3f}:({top5.avg:.3f})'.format(
                          batch_idx,
                          len(test_loader),
                          args.world_size * args.batch_size / batch_time.val,
                          args.world_size * args.batch_size / batch_time.avg,
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))
                
    print('Prec@1={top1.avg:.3f}, Prec@5={top5.avg:.3f}'.format(top1=top1,
                                                                 top5=top5))

def check_sagemaker(args):
    ## SageMaker
    if os.environ.get('SM_MODEL_DIR') is not None:
        args.data_path = os.environ['SM_CHANNEL_TRAINING']
        args.save_model =os.environ.get('SM_MODEL_DIR')
    return args

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/tmp/data",
        help="Path for downloading " "the MNIST dataset",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=10000,
    )
    parser.add_argument('--mp_parameters', type=str, default='')
    parser.add_argument('--prescaled_batch', type=lambda s:s.lower() in ['true', 't', 'yes','1'], default=False)
    parser.add_argument('--smp', type=lambda s:s.lower() in ['true', 't', 'yes','1'], default=False)
    parser.add_argument('--ddp', type=lambda s:s.lower() in ['true', 't', 'yes','1'], default=False)
    
    ########################################################
    ####### 2. SageMaker Distributed Data Parallel   #######
    #######  - Get all number of GPU and rank number #######
    ########################################################
    
    args = parser.parse_args()
    
    args = check_sagemaker(args)
    
    if args.smp:
        smp.init()
        args.rank = rank = smp.dp_rank() if not args.prescaled_batch else smp.rdp_rank()
        args.world_size = smp.dp_size() if not args.prescaled_batch else smp.rdp_size()
        args.local_rank = local_rank = smp.local_rank()
        drop_last = True
        
    else:
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.rank = rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.local_rank = local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=args.world_size)
        
        drop_last = False
    ########################################################
    

#         args.batch_size //= args.world_size // 8
#         args.batch_size = max(args.batch_size, 1)
    args.lr = 0.0001 #0.001
    data_path = args.data_path


    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )

    if not torch.cuda.is_available():
        raise CUDANotFoundException(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # select a single rank per node to download data
    train_dataset = datasets.ImageFolder(
        root=data_path+'/train',
        transform=transforms.Compose(
            [transforms.ToTensor(),
#              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Resize((args.img_size,args.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
             transforms.Normalize(
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
             )
            ]
        ),
    )

    #######################################################
    ####### 3. SageMaker Distributed Data Parallel  #######
    #######  - Add num_replicas and rank            #######
    #######################################################
    if args.smp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            shuffle=True,
            num_replicas=args.world_size, 
            rank=args.rank,
            drop_last=True,
        )

#     if args.smp and args.ddp and smp.dp_size() > 1:
#         print("DDP SMP")
#         partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
#         train_dataset = SplitDataset(train_dataset, partitions=partitions_dict)
#         train_dataset.select(f"{smp.dp_rank()}")
        
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank
        )
    #######################################################
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=data_path+'/val',
            transform=transforms.Compose(
                [transforms.ToTensor(), 
#                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                 transforms.Resize((args.img_size,args.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                 transforms.Normalize(
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225],
                 )
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=drop_last
    )

    #######################################################
    ####### 4. SageMaker Distributed Data Parallel  #######
    #######  - Add num_replicas and rank            #######
    ####################################################### 
    torch.cuda.set_device(local_rank)
    
    # choose model from pytorch model_zoo
    model = util.torch_model(
        'vgg11',
        num_classes=37,
        pretrained=True,
        local_rank=args.local_rank,
        model_parallel=args.smp)  # 1000 resnext101_32x8d
    
    args.criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.smp:
        scaler = smp.amp.GradScaler()
    else:
        scaler = None
    
    if args.smp:
        model = smp.DistributedModel(model, trace_device="gpu")
        optimizer = smp.DistributedOptimizer(optimizer)
    else:
        model = DDP(model.to(device),
                    device_ids=[local_rank],
                    output_device=local_rank)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):        
        train(args, model, device, train_loader, optimizer, epoch, scaler)
        test(args, model, device, test_loader)
        scheduler.step()

    
    if args.save_model:
        if args.smp:
            if smp.dp_rank() == 0:
                model_dict = model.local_state_dict()
                opt_dict = optimizer.local_state_dict()
                smp.save(
                    {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                    args.save_model + "/model.pt",
                    partial=True
                )
            smp.barrier()
        else:
            torch.save(model.state_dict(), args.save_model + "/model.pt")
        
        


if __name__ == "__main__":
    main()
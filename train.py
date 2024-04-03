## Author : Metz copy from Bertrand CABOT from IDRIS(CNRS)
# **************************************************************************************************************
import argparse  #
import contextlib  #
import os  #
import random  #

####       DON'T MODIFY    ####################        #
import numpy as np
import torch  #
import torchvision
import torchvision.transforms as transforms  #
import wandb
from torch.utils.checkpoint import checkpoint_sequential  #

from dlojz_chrono import Chronometer  #
from orthog_model import *

random.seed(123)  #
np.random.seed(123)  #
torch.manual_seed(123)  #
# **************************************************************************************************************

import torch.distributed as dist
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
import idr_torch  #
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy

VAL_BATCH_SIZE = 256


# **************************************************************************************************************
def train():  #
    parser = argparse.ArgumentParser()  #
    parser.add_argument('-b', '--batch-size', default=128, type=int,  #
                        help='batch size per GPU')  #
    parser.add_argument('-e', '--epochs', default=1, type=int,  #
                        help='number of total epochs to run')  #
    parser.add_argument('--image-size', default=224, type=int,  #
                        help='Image size')  # #
    parser.add_argument('--test', default=False, action='store_true',  ##    DON'T MODIFY    ########      #
                        help='Test 50 iterations')  #
    parser.add_argument('--test-nsteps', default='50', type=int,  #
                        help='the number of steps in test mode')  #
    parser.add_argument('--num-workers', default=10, type=int,  #
                        help='num workers in dataloader')  #
    parser.add_argument('--persistent-workers', default=True, action=argparse.BooleanOptionalAction,  #
                        help='activate persistent workers in dataloader')  #
    parser.add_argument('--pin-memory', default=True, action=argparse.BooleanOptionalAction,  #
                        help='activate pin memory option in dataloader')  #
    parser.add_argument('--non-blocking', default=True, action=argparse.BooleanOptionalAction,  #
                        help='activate asynchronuous GPU transfer')  #
    parser.add_argument('--prefetch-factor', default=3, type=int,  #
                        help='model name')
    parser.add_argument('--model-name', default="vits16", type=str,  #
                        help='prefectch factor in dataloader')  #
    parser.add_argument('--drop-last', default=False, action=argparse.BooleanOptionalAction,  #
                        help='activate drop_last option in dataloader')  #
    # **************************************************************************************************************

    ## Add parser arguments
    parser.add_argument('--prof', default=False, action='store_true', help='PROF implementation')

    args = parser.parse_args()

    ## chronometer initialisation (test and rank)
    chrono = Chronometer(args.test, idr_torch.rank)  ### DON'T MODIFY ###

    # configure distribution method: define rank and initialise communication backend (NCCL)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=idr_torch.size, rank=idr_torch.rank)

    # define model
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")

    model = orthog_model(orthog=True, model_name=args.model_name)
    model = model.to(gpu)

    # **************************************************************************************************************
    if idr_torch.rank == 0: print(f'model: {args.model_name}')  #
    if idr_torch.rank == 0: print('number of parameters: {}'.format(sum([p.numel()  ### DON'T MODIFY ####      #
                                                                         for p in model.parameters()])))  #
    # *************************************************************************************************************#

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=always_wrap_policy,
        mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float32),
    )

    # *************************************************************************************************************#
    # distribute batch size (mini-batch)                                                                      #
    num_replica = idr_torch.size  ### DON'T MODIFY ##################       #
    mini_batch_size = args.batch_size  #
    global_batch_size = mini_batch_size * num_replica  #
    #
    if idr_torch.rank == 0:  #
        print(f'global batch size: {global_batch_size} - mini batch size: {mini_batch_size}')  #
        #
    # **************************************************************************************************************

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # **************************************************************************************************************
    # create scheduler
    # cosine learning rate schedule with a linear warmup (10k steps) with optimizer adamw (b1 : 0.9, b2 : 0.999) and clip the gradient to 1 :

    # Create the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=10_000, pct_start=0.1,
                                                    anneal_strategy='cos')

    if idr_torch.rank == 0: print(f'Optimizer: {optimizer}')  ### DON'T MODIFY ###

    #########  DATALOADER ############
    # Define a transform to pre-process the training images.

    if idr_torch.rank == 0: print(
        f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ")  ### DON'T MODIFY ###

    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),  # Random resize - Data Augmentation
        transforms.RandomHorizontalFlip(),  # Horizontal Flip - Data Augmentation
        transforms.RandAugment(2, 9),  # RandAugment - Data Augmentation
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH'] + '/imagenet', split="train",
                                                  transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank,
                                                                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=mini_batch_size,
                                               shuffle=False,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               persistent_workers=args.persistent_workers,
                                               pin_memory=args.pin_memory,
                                               prefetch_factor=args.prefetch_factor,
                                               drop_last=args.drop_last)

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

    val_dataset = torchvision.datasets.ImageNet(root=os.environ['ALL_CCFRSCRATCH'] + '/imagenet', split='val',
                                                transform=val_transform)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                  num_replicas=idr_torch.size,
                                                                  rank=idr_torch.rank,
                                                                  shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             prefetch_factor=args.prefetch_factor,
                                             drop_last=args.drop_last)

    N_batch = len(train_loader)
    N_val_batch = len(val_loader)
    N_val = len(val_dataset)

    chrono.start()  ### DON'T MODIFY ####

    ## Initialisation
    if idr_torch.rank == 0: accuracies = []
    val_loss = torch.Tensor([0.]).to(gpu)  # send to GPU
    val_accuracy = torch.Tensor([0.]).to(gpu)  # send to GPU

    # Pytorch profiler setup
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=1, warmup=1, active=12, repeat=1),
                   on_trace_ready=tensorboard_trace_handler('./profiler/' + os.environ['SLURM_JOB_NAME']
                                                            + '_' + os.environ['SLURM_JOBID'] + '_bs' +
                                                            str(mini_batch_size) + '_is' + str(args.image_size)),
                   profile_memory=True,
                   record_shapes=False,
                   with_stack=False,
                   with_flops=False
                   ) if args.prof else contextlib.nullcontext()

    # **************************************************************************************************************
    ### Weight and biases initialization                                                                      #
    if not args.test and idr_torch.rank == 0:  #
        config = dict(  #
            architecture=args.model_name,  #
            batch_size=args.batch_size,  #
            epochs=args.epochs,  #
            image_size=args.image_size,  #
            learning_rate=args.lr,  #
            weight_decay=args.wd,  #
            momentum=args.mom,  #
            optimizer=optimizer.__class__.__name__,  #
            lr_scheduler=scheduler.__class__.__name__  #### DON'T MODIFY ######           #
        )  #
        #
        wandb.init(  #
            project="Imagenet Race Cup",  #
            entity="test_vit",  #
            name=os.environ['SLURM_JOB_NAME'] + '_' + os.environ['SLURM_JOBID'],  #
            tags=['label smoothing'],  #
            config=config,  #
            mode='offline'  #
        )  #
        wandb.watch(model, log="all", log_freq=N_batch)  #
    # **************************************************************************************************************

    #### TRAINING ############
    with prof:
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            # **************************************************************************************************************
            chrono.dataload()  #
            if idr_torch.rank == 0: chrono.tac_time(clear=True)  #
            #
            for i, (images, labels) in enumerate(train_loader):  #
                ### DON'T MODIFY ##############          #
                csteps = i + 1 + epoch * N_batch
                if args.test: print(f'Train step {csteps} - rank {idr_torch.rank}')  #
                if args.test and csteps > args.test_nsteps: break  #
                if i == 0 and idr_torch.rank == 0:  #
                    print(f'image batch shape : {images.size()}')  #
                    #
                # **************************************************************************************************************

                # distribution of images and labels to all GPUs
                images = images.to(gpu, non_blocking=args.non_blocking)
                labels = labels.to(gpu, non_blocking=args.non_blocking)

                # **************************************************************************************************************
                chrono.dataload()  #
                chrono.training()  ### DON'T MODIFY #################                                #
                chrono.forward()  #
                # **************************************************************************************************************

                optimizer.zero_grad()
                # Runs the forward pass
                outputs = model(images)[:, 0]
                loss = criterion(outputs, labels)

                # **************************************************************************************************************
                chrono.forward()  #
                chrono.backward()  ### DON'T MODIFY ###############                                #
                # **************************************************************************************************************

                optimizer.step()

                # Metric mesurement
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).sum() / labels.size(0)
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                accuracy /= idr_torch.size
                if idr_torch.rank == 0: accuracies.append(accuracy.item())

                # ***************************************************************************************************************
                if not args.test and idr_torch.rank == 0 and csteps % 10 == 0:  #
                    wandb.log({"train accuracy": accuracy.item(),  #
                               "train loss": loss.item(),  #
                               "learning rate": scheduler.get_lr()[0]}, step=csteps)  #
                    #
                chrono.backward()  #
                chrono.training()  ### DON'T MODIFY ###########              #
                #
                #
                if ((i + 1) % (N_batch // 10) == 0 or i == N_batch - 1) and idr_torch.rank == 0:  #
                    print('Epoch [{}/{}], Step [{}/{}], Time: {:.3f}, Loss: {:.4f}, Acc:{:.4f}'.format(  #
                        epoch + 1, args.epochs, i + 1, N_batch,  #
                        chrono.tac_time(), loss.item(), np.mean(accuracies)))  #
                    #
                    accuracies = []  #
                # ***************************************************************************************************************

                # scheduler update
                scheduler.step()

                # profiler update
                if args.prof: prof.step()

                # ***************************************************************************************************************
                chrono.dataload()  #
                #
                #### VALIDATION ############                                                                       #
                if ((i == N_batch - 1) or (args.test and i == args.test_nsteps - 1)):  #
                    #
                    chrono.validation()  #
                    model.eval()  ### DON'T MODIFY ############              #
                    if args.test: print(f'Train step 100 - rank {idr_torch.rank}')  #
                    for iv, (val_images, val_labels) in enumerate(val_loader):  #
                        # ***************************************************************************************************************

                        # distribution of images and labels to all GPUs
                        val_images = val_images.to(gpu, non_blocking=args.non_blocking)
                        val_labels = val_labels.to(gpu, non_blocking=args.non_blocking)

                        # Runs the forward pass with no grade mode.
                        with torch.no_grad():
                            val_outputs = model(val_images)
                            loss = criterion(val_outputs, val_labels)

                        # ***************************************************************************************************************
                        val_loss += (loss * val_images.size(0) / N_val)  #
                        _, predicted = torch.max(val_outputs.data, 1)  #
                        val_accuracy += ((predicted == val_labels).sum() / N_val)  ### DON'T MODIFY #######        #
                        #
                        if args.test and iv >= 20: break  #
                    # ***************************************************************************************************************

                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)

                    # ***************************************************************************************************************
                    model.train()  #
                    chrono.validation()  #
                    if idr_torch.rank == 0: assert val_accuracy.item() <= 1., 'Something wrong with your allreduce'  #
                    if not args.test and idr_torch.rank == 0:  ### DON'T MODIFY #############    #
                        print('##EVALUATION STEP##')  #
                        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(  #
                            epoch + 1, args.epochs, val_loss.item(), val_accuracy.item()))  #
                        print(">>> Validation complete in: " + str(chrono.val_time))  #
                        if val_accuracy.item() > 1.:  #
                            print('ddp implementation error : accuracy outlier !!')  #
                            wandb.log({"test accuracy": None,  #
                                       "test loss": val_loss.item()})  #
                        else:  #
                            wandb.log({"test accuracy": val_accuracy.item(),  #
                                       "test loss": val_loss.item()})  #
                    # ***************************************************************************************************************

                    ## Clear validations metrics
                    val_loss -= val_loss
                    val_accuracy -= val_accuracy

    ## Be sure all process finish at the same time to avoid incoherent logs at the end of process
    dist.barrier()

    # ***************************************************************************************************************                                                                                                                                       #
    chrono.display(N_val_batch)  #
    if idr_torch.rank == 0:  ### DON'T MODIFY ###############              #
        print(">>> Number of batch per epoch: {}".format(N_batch))  #
        print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes')  #
    else:  #
        print(f'MaxMemory for GPU:{idr_torch.rank} {torch.cuda.max_memory_allocated()} Bytes')  #
    # ***************************************************************************************************************

    # Save last checkpoint
    if not args.test and idr_torch.rank == 0:
        checkpoint_path = f"checkpoints/{os.environ['SLURM_JOBID']}_{global_batch_size}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print("Last epoch checkpointed to " + checkpoint_path)


if __name__ == '__main__':
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes")
    train()


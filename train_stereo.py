from __future__ import print_function, division
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from core.crestereo import CREStereo_plusplus
import core.stereo_datasets as datasets
from torch.utils.data.distributed import DistributedSampler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()

    epe = epe.view(-1) * valid.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100
    def __init__(self, scheduler, log_path):
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=log_path)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = f"[{self.total_steps+1:6d}, {self.scheduler.get_last_lr()[0]:10.7f}] "
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            self.running_loss[key] = self.running_loss.get(key, 0.0) + metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def setup_ddp():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_count = torch.cuda.device_count()
    assert 0 <= local_rank < device_count, f"Invalid local_rank={local_rank}, only {device_count} CUDA devices visible"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    dist.barrier(device_ids=[local_rank])
    return local_rank, rank

def train(args):
    local_rank, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # DDP-safe seed
    torch.manual_seed(666 + rank)
    np.random.seed(666 + rank)

    model = CREStereo_plusplus().to(device)
    if rank == 0:
        print("Parameter Count:", count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(scheduler, args.log_path) if rank == 0 else None

    if args.restore_ckpt:
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(args.restore_ckpt, map_location=map_location)
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    validation_frequency = args.validation_frequency
    total_steps = 0
    global_batch_num = 0

    while total_steps <= args.num_steps:
        model.train(True)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(total_steps)

        iterator = tqdm(train_loader) if rank == 0 else train_loader
        for _, *data_blob in iterator:
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]

            disp_preds = model(image1, image2)

            disp_gt = torch.cat([disp_gt, disp_gt * 0], dim=1)

            loss, metrics = sequence_loss(disp_preds, disp_gt, valid)

            if logger:
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
                logger.writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            total_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if logger:
                logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1 and rank == 0:
                save_path = Path(args.ckpt_path) / f"{total_steps+1}_{args.name}.pth"
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.module.state_dict(), save_path)

            if total_steps > args.num_steps:
                break

        process = psutil.Process(os.getpid())
        mem_MB = process.memory_info().rss / 1024 / 1024
        logging.info(f"[MEM] Step {total_steps}: {mem_MB:.1f} MB")
        gc.collect()


    if rank == 0:
        logger.close()
        torch.save(model.module.state_dict(), Path(args.ckpt_path) / f"{args.name}.pth")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='crestereo')
    parser.add_argument('--restore_ckpt', default=None)
    parser.add_argument('--ckpt_path', default='./checkpoints/sceneflow')
    parser.add_argument('--log_path', default='./checkpoints/sceneflow')
    parser.add_argument('--validation_frequency', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None)
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4])
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'])
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8])
    parser.add_argument('--noyjitter', action='store_true')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    train(args)
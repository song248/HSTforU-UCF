import os
import time
import json
import random
import inspect
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from timm.utils import AverageMeter

# configs
from configs.default import get_config  # ✅ 원본과 동일한 위치에서 get_config 사용:contentReference[oaicite:1]{index=1}
# datasets
from datasets.build_dataset import build_train_loader  # ✅ 원본 데이터로더 사용:contentReference[oaicite:2]{index=2}
# models
import models
# utils
from utils.logger import create_logger
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from utils.criterion import Losses
from utils.anomaly_score import psnr_park
from utils.checkpoint import save_checkpoint


def parse_option():
    parser = argparse.ArgumentParser('HSTforU training script (AMP-enabled)', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", default=None, nargs='+',
                        help="Modify config options by adding 'KEY VALUE' pairs.")
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--gpus', type=int, help='GPU index')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config


def main(config, logger):
    dataset_train, data_loader_train = build_train_loader(config)
    logger.info(f"The number of input sequences of {config.DATA.DATASET.upper()} dataset: {len(dataset_train)}")

    logger.info(f"-------------------------------------------------")
    logger.info(f"{inspect.getsourcefile(models.build_model)}")
    logger.info(f"-------------------------------------------------")

    model = models.build_model(config, logger=logger)

    # (옵션) 사전학습 체크포인트 로드: ckpt/ckpt_shanghaitech.pth 또는 pvt_v2_b2.pth
    if os.path.isfile(config.MODEL.PRETRAINED):
        logger.info(f"Loading fine-tuning checkpoint from {config.MODEL.PRETRAINED}")
        checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        # DDP/Lightning 저장물 prefix 정리
        new_state = { (k.replace("module.", "") if k.startswith("module.") else k): v
                      for k, v in state_dict.items() }
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        logger.info(f"Loaded checkpoint. Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        logger.warning(f"No checkpoint found at {config.MODEL.PRETRAINED}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters:,}")

    model.cuda()
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # ✅ AMP & Scaler
    scaler = GradScaler()

    # 원본 손실 조합 유지 (intensity + gradient + MS-SSIM + L2):contentReference[oaicite:3]{index=3}
    criterion = Losses(config).cuda()
    mse = torch.nn.MSELoss(reduction='none')

    # ✅ Gradient Accumulation (환경변수로 제어: 기본 1)
    accum_steps = int(os.environ.get("ACCUM_STEPS", "1"))
    if accum_steps > 1:
        logger.info(f"Using Gradient Accumulation: ACCUM_STEPS={accum_steps}")

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 분산 샘플러 epoch 설정:contentReference[oaicite:4]{index=4}
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train,
                        optimizer, epoch, lr_scheduler, mse, logger,
                        scaler=scaler, accum_steps=accum_steps)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, logger, resume=False)

    total_time = time.time() - start_time
    logger.info(f'Training time {str(datetime.timedelta(seconds=int(total_time)))}')


def train_one_epoch(config, model, criterion, data_loader, optimizer,
                    epoch, lr_scheduler, mse, logger, scaler: GradScaler, accum_steps: int = 1):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    optimizer.zero_grad(set_to_none=True)

    for idx, samples in enumerate(data_loader):
        # 원본 데이터 포맷 유지: dict['video']에서 마지막 프레임 target 분리:contentReference[oaicite:5]{index=5}
        videos = samples['video']
        inputs, targets = videos[:-1], videos[-1]
        targets = targets.cuda(non_blocking=True)

        # ✅ AMP 구간
        with autocast():
            outputs = model(inputs)
            int_l, gra_l, mss_l, l2_l = criterion(outputs, targets)
            loss = (int_l + gra_l + mss_l + l2_l)

        # ✅ Gradient Accumulation
        if accum_steps > 1:
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()
        else:
            scaler.scale(loss).backward()

        # step/update 조건
        if (idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 스케줄러는 step 기반 유지:contentReference[oaicite:6]{index=6}
        lr_scheduler.step_update(epoch * num_steps + idx)

        # compute PSNR (로그용)
        with torch.no_grad():
            mse_imgs = torch.mean(mse((outputs + 1) / 2, (targets + 1) / 2)).item()
            psnr = psnr_park(mse_imgs)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'lr {lr:.6f}\t'
                f'psnr {psnr:.2f}\t'
                f'int_l {int_l:.6f}\t'
                f'gra_l {gra_l:.6f}\t'
                f'mss_l {mss_l:.6f}\t'
                f'l2_l {l2_l:.6f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    # torchrun이 넘겨주는 LOCAL_RANK 반영:contentReference[oaicite:7]{index=7}
    if "LOCAL_RANK" in os.environ:
        lr_env = int(os.environ["LOCAL_RANK"])
        config.defrost()
        config.LOCAL_RANK = lr_env
        config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # 시드/벤치마크 설정:contentReference[oaicite:8]{index=8}
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, logger)

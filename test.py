import argparse, json, os, numpy as np, torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm   # ★ 추가

import models
from configs.default import get_config
from datasets.build_dataset import build_test_loader
from utils.logger import create_logger
from utils.anomaly_score import psnr_park


def parse_option():
    parser = argparse.ArgumentParser('HSTforU test script (UCF-Crime subset)', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/scripts/ucfcrime.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--pretrained', type=str, required=True, help='path to trained checkpoint')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--test-list', type=str, required=True, help="path to Anomaly_Test.txt")
    parser.add_argument('--anno-file', type=str, required=True,
                        help="path to Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    args, _ = parser.parse_known_args()
    config = get_config(args)

    if args.pretrained:
        config.defrost()
        config.MODEL.PRETRAINED = args.pretrained
        config.freeze()
    return args, config


def load_test_list(list_path):
    with open(list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_ucfcrime_annotations(anno_path, frames_root):
    anno = {}
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[0]
            name_wo_ext = os.path.splitext(name)[0]
            nums = [int(x) for x in parts[2:]]
            pairs = []
            for i in range(0, len(nums), 2):
                if i + 1 < len(nums):
                    s, e = nums[i], nums[i + 1]
                    if s >= 0 and e >= 0 and e >= s:
                        pairs.append((s, e))
            anno[name_wo_ext] = pairs

    labels = {}
    for folder in sorted(os.listdir(frames_root)):
        vid = folder
        vf = os.path.join(frames_root, folder)
        length = len([p for p in os.listdir(vf) if not p.startswith('.')])
        sub_gt = np.zeros((length,), dtype=np.int8)
        if vid in anno:
            for (s, e) in anno[vid]:
                s_idx = max(s - 1, 0)
                e_idx = min(e, length)
                sub_gt[s_idx:e_idx] = 1
        labels[vid] = sub_gt
    return labels


def validate(config, model, logger, args):
    logger.info("Start testing...")

    dataset_test, data_loader_test = build_test_loader(config)
    logger.info(f"Test dataset: {config.DATA.DATASET}, length {len(dataset_test)}")

    target_list = load_test_list(args.test_list)
    allowed = set(Path(v).stem for v in target_list)
    logger.info(f"Loaded {len(target_list)} test videos from {args.test_list}")

    labels_dict = load_ucfcrime_annotations(
        args.anno_file,
        os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, 'testing', 'frames')
    )

    model.eval()
    scores, gts = [], []

    with torch.no_grad():
        # ★ tqdm 추가 (전체 길이 설정)
        for batch in tqdm(data_loader_test, desc="Testing", total=len(data_loader_test)):
            frames = batch["frames"].cuda(non_blocking=True)   # (B, T, C, H, W)
            vid_names = batch["video_name"]
            frame_idxs = batch["frame_idx"]

            B, T, C, H, W = frames.shape
            x_list = [frames[:, t] for t in range(T - 1)]
            target = frames[:, -1]
            outputs = model(x_list)

            mse_imgs = torch.mean((outputs - target) ** 2, dim=[1, 2, 3]).cpu().numpy()

            for b, mse_val in enumerate(mse_imgs):
                vid_name = vid_names[b]
                if vid_name not in allowed:
                    continue

                psnr = psnr_park(mse_val)
                scores.append(psnr)

                frame_idx = int(frame_idxs[b])
                if vid_name in labels_dict:
                    gts.append(int(labels_dict[vid_name][frame_idx]))
                else:
                    gts.append(0)

    scores = np.array(scores, dtype=np.float32)
    gts = np.array(gts, dtype=np.int32)
    smin, smax = scores.min(), scores.max()
    scores = (scores - smin) / (smax - smin + 1e-8)

    auc = roc_auc_score(gts, scores)
    logger.info(f"Evaluation AUC: {auc:.4f}")
    return auc


def main(config, logger, args):
    model = models.build_model(config, logger=logger)

    ckpt_path = config.MODEL.PRETRAINED
    assert os.path.isfile(ckpt_path), f"{ckpt_path} not found!"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    new_state = { (k.replace("module.", "") if k.startswith("module.") else k): v
                  for k, v in state_dict.items() }
    model.load_state_dict(new_state, strict=False)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )

    auc = validate(config, model, logger, args)
    logger.info(f"Final AUC on {config.DATA.DATASET}: {auc:.4f}")


if __name__ == '__main__':
    args, config = parse_option()

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
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()

    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config_test.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, logger, args)

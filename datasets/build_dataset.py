import os
import natsort
from PIL import Image
from random import randrange

import torch
import torch.distributed as dist
import torch.utils.data as data
from torchvision import transforms
from timm.data import create_transform

try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_transform(is_train, config):
    if is_train:
        transform = create_transform(
            input_size=(config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            is_training=True,
            hflip=config.AUG.HFLIP,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation=config.DATA.INTERPOLATION,
        )
        transform.transforms[0] = transforms.Resize(
            (config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            interpolation=_pil_interp(config.DATA.INTERPOLATION)
        )
        return transform
    else:
        t = [
            transforms.Resize((config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
                              interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transforms.Compose(t)


def collect_files(root):
    include_ext = [".png", ".jpg", "jpeg", ".bmp"]
    dirs = [x[0] for x in os.walk(root, followlinks=True)]
    dirs = natsort.natsorted(dirs)

    dataset = [
        [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
         if os.path.isfile(os.path.join(fdir, el))
         and not el.startswith('.')
         and any([el.lower().endswith(ext) for ext in include_ext])]
        for fdir in dirs
    ]
    return [el for el in dataset if el]


def build_indexes(videos):
    indexes = []
    start, end = 0, -1
    for video in videos:
        start = end + 1
        end = start + len(video) - 1
        indexes.append([start, end])
    return indexes


def get_video_index(indexes, ind):
    i = -1
    for i, index in enumerate(indexes):
        if index[0] <= ind <= index[-1]:
            return i
    return i


class BuildTestDataset(data.Dataset):
    def __init__(self, config):
        super(BuildTestDataset, self).__init__()
        self.transform = build_transform(is_train=False, config=config)

        # UCF-Crime-frames/ucfcrime/testing/frames (사용자 경로 규약)
        dir_path = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET, 'testing', "frames")
        if config.DATA.DATASET_SCENE != '':
            dir_path = os.path.join(dir_path, config.DATA.DATASET_SCENE)
        assert os.path.exists(dir_path), f"Testing path not found: {dir_path}"

        self.videos = collect_files(dir_path)

        self.num_input_frames = config.DATA.NUM_INPUT_FRAMES  # 보통 5 (4 입력 + 1 타깃):contentReference[oaicite:3]{index=3}
        self.cut_videos = []
        self.num_videos = 0
        for video in self.videos:
            self.num_videos += (len(video) - (self.num_input_frames - 1))
            cut_video = [video[i] for i in range(len(video) - (self.num_input_frames - 1))]
            self.cut_videos.append(cut_video)

        self.indexes = build_indexes(self.cut_videos)

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        i_video = get_video_index(self.indexes, index)
        idx = index - self.indexes[i_video][0]
        video = self.videos[i_video]

        frames = []
        for i in range(self.num_input_frames):
            frame = Image.open(video[idx + i]).convert('RGB')
            frames.append(self.transform(frame))  # (C,H,W)

        # 비디오 폴더 이름 (정답 매칭용: annotation key와 동일하게 사용)
        video_name = os.path.basename(os.path.dirname(video[0]))

        return {
            # 여기서는 (T,C,H,W) 로 반환 → DataLoader 가 (B,T,C,H,W) 로 스택
            "frames": torch.stack(frames, dim=0),
            "i_video": i_video,
            "frame_idx": idx,
            "video_name": video_name
        }


def build_test_loader(config):
    dataset_test = BuildTestDataset(config=config)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset_test, data_loader_test

import os
import glob
import numpy as np
import scipy.io as scio


class Label:
    def __init__(self, config):
        root = config.DATA.DATA_PATH
        dataset_name = config.DATA.DATASET.lower()
        scene = config.DATA.DATASET_SCENE
        self.dataset_name = dataset_name

        # datasets with pre-generated frame-wise masks (numpy) -----------------
        if dataset_name == 'shanghaitech':
            self.frame_mask = os.path.join(root, 'shanghaitech', 'test_frame_mask/*')
        elif dataset_name == 'drone':
            self.frame_mask = os.path.join(root, 'drone', f'annotation/{scene}/*')
        elif dataset_name in ['drone/railway', 'drone/highway', 'drone/crossroads',
                              'drone/bike', 'drone/vehicle', 'drone/solar', 'drone/farmland']:
            self.frame_mask = os.path.join(root, dataset_name, 'annotation/*')
        else:
            self.frame_mask = None

        # matlab-style gt (avenue/ped2)
        self.mat_path = os.path.join(root, dataset_name, dataset_name + '.mat')

        # test video folders (in order used by loader)
        test_dataset_path = os.path.join(root, dataset_name, "testing", "frames")
        video_folders = (os.listdir(test_dataset_path))
        video_folders.sort()
        self.video_folders = [os.path.join(test_dataset_path, folder) for folder in video_folders]

        # UCF-Crime temporal annotation file
        if dataset_name == 'ucfcrime':
            self.ucf_anno = os.path.join(root, 'ucfcrime', 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt')

    def _basename_wo_ext(self, p):
        bn = os.path.basename(p)
        return os.path.splitext(bn)[0]

    def __call__(self):
        # pre-generated frame masks path (ShanghaiTech / Drone*)
        if self.dataset_name == 'shanghaitech' or self.dataset_name == 'drone' or \
           self.dataset_name in ['drone/railway', 'drone/highway', 'drone/crossroads',
                                 'drone/bike', 'drone/vehicle', 'drone/solar', 'drone/farmland']:
            np_list = glob.glob(self.frame_mask)
            np_list.sort()
            gt = [np.load(npy) for npy in np_list]
            return gt

        # UCF-Crime: parse temporal annotation txt -> frame-level gt ----------
        if self.dataset_name == 'ucfcrime':
            # Build dict: video_name(without .mp4) -> list of (start, end) 1-based inclusive
            anno = {}
            with open(self.ucf_anno, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Expected: name, class, s1, e1, s2, e2  (s2/e2 may be -1)
                    name = parts[0]
                    name_wo_ext = os.path.splitext(name)[0]
                    # robust parse: keep all integer pairs after the 2nd column
                    nums = [int(x) for x in parts[2:]]
                    pairs = []
                    for i in range(0, len(nums), 2):
                        if i + 1 < len(nums):
                            s, e = nums[i], nums[i + 1]
                            if s >= 0 and e >= 0 and e >= s:
                                pairs.append((s, e))
                    anno[name_wo_ext] = pairs

            all_gt = []
            for vf in self.video_folders:
                vid = self._basename_wo_ext(vf)  # folder name
                length = len([p for p in os.listdir(vf) if not p.startswith('.')])
                sub_video_gt = np.zeros((length,), dtype=np.int8)

                if vid in anno:
                    for (s, e) in anno[vid]:
                        # txt is 1-based; codebase uses start-1:end
                        s_idx = max(s - 1, 0)
                        e_idx = min(e, length)
                        sub_video_gt[s_idx:e_idx] = 1
                # if not in anno -> normal video -> keep zeros
                all_gt.append(sub_video_gt)
            return all_gt

        # Avenue/Ped2 (mat file with intervals) --------------------------------
        abnormal_mat = scio.loadmat(self.mat_path, squeeze_me=True)['gt']
        all_gt = []
        for i in range(abnormal_mat.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_mat[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start:end] = 1
            all_gt.append(sub_video_gt)
        return all_gt

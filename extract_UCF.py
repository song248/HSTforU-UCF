import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def ffmpeg_extract(video_file, out_dir, fps, img_size, gpu_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")):
        return str(out_dir), "skipped"

    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-hwaccel_device", str(gpu_id),
        "-i", str(video_file),
        "-vf", f"fps={fps},scale={img_size}:{img_size}",
        "-q:v", "5",   # 품질 5 (적당히 손실 압축, 용량 절약)
        str(out_dir / "%06d.jpg"),
        "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
        return str(out_dir), "done"
    except subprocess.CalledProcessError as e:
        return str(video_file), f"error: {e}"

def extract_ucf_frames_parallel(
    src_root="/home/song/Desktop/HiTESS/data/UCF-Crime",
    train_list="Anomaly_Train.txt",
    test_list="Anomaly_Test.txt",
    dst_root="./UCF-Crime-frames",
    fps=10,              # 10fps로 변경
    img_size=224,        # 224로 리사이즈
    num_gpus=2,
    workers_per_gpu=2
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    train_dst = dst_root / "training" / "frames"
    test_dst = dst_root / "testing" / "frames"
    train_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)

    def read_list(txt_path):
        with open(txt_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_videos = read_list(train_list)
    test_videos = read_list(test_list)

    all_jobs = []
    for rel_path in train_videos:
        video_file = src_root / rel_path
        vid_name = Path(video_file).stem
        out_dir = train_dst / vid_name
        all_jobs.append((video_file, out_dir, "train"))
    for rel_path in test_videos:
        video_file = src_root / rel_path
        vid_name = Path(video_file).stem
        out_dir = test_dst / vid_name
        all_jobs.append((video_file, out_dir, "test"))

    # 멀티스레드 실행 (GPU round-robin 할당)
    results = []
    with ThreadPoolExecutor(max_workers=num_gpus * workers_per_gpu) as executor:
        future_to_job = {}
        for idx, (video_file, out_dir, split) in enumerate(all_jobs):
            gpu_id = idx % num_gpus
            future = executor.submit(ffmpeg_extract, video_file, out_dir, fps, img_size, gpu_id)
            future_to_job[future] = video_file

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Extracting all videos"):
            res = future.result()
            results.append(res)

    print("[DONE] Extraction complete.")
    return results

if __name__ == "__main__":
    extract_ucf_frames_parallel(
        src_root="/home/song/Desktop/HiTESS/data/UCF-Crime",
        train_list="Anomaly_Train.txt",
        test_list="Anomaly_Test.txt",
        dst_root="./UCF-Crime-frames",
        fps=10,           # 요구사항 반영
        img_size=224,     # 요구사항 반영
        num_gpus=2,       # A4000 두 장 사용
        workers_per_gpu=2
    )

import os
import pickle
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
from tqdm import tqdm

from model_clip import AVEmotionCLIPModel


# =========================================================
# 1. Config
# =========================================================
@dataclass
class InferConfig:
    test_pkl: str = "/media/SSD/data/CVPR_workshop/pkl_index/test_real.pkl"
    checkpoint_path: str = "/home/jbj/cvpr_workshop/jbj_test/result/checkpoints_tcn_fusion/best_clip_model.pt"

    output_dir: str = "/media/SSD/data/CVPR_workshop/test_result"
    output_txt_name: str = "result.txt"

    image_size: int = 224
    num_image_frames: int = 20
    window_sec: float = 10.0

    target_sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024

    batch_size: int = 64
    num_workers: int = 8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Clamp outputs to the range [-1, 1] before saving results
    clamp_output: bool = True


# =========================================================
# 2. Utils
# =========================================================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def clamp_va(x: float) -> float:
    return max(-1.0, min(1.0, x))


def sample_uniform_frames_from_list(frame_list: List[int], num_samples: int) -> List[int]:
    """
    Uniformly sample num_samples frames from frame_list.
    If the list is shorter than required, duplication is allowed.
    """
    if len(frame_list) == 0:
        raise ValueError("frame_list is empty")

    if len(frame_list) == 1:
        return [frame_list[0]] * num_samples

    idxs = torch.linspace(0, len(frame_list) - 1, steps=num_samples)
    idxs = torch.round(idxs).long().tolist()
    return [frame_list[i] for i in idxs]


# =========================================================
# 3. Video metadata builder
# =========================================================
def build_video_metadata(test_pkl: str) -> OrderedDict:
    """
    Build per-video metadata from pkl['samples'].
    Scan the actual image folders and use all existing jpg files as inference targets.
    """
    pkl_data = load_pickle(test_pkl)

    if not isinstance(pkl_data, dict):
        raise TypeError(f"Expected dict pickle, got {type(pkl_data)}")
    if "samples" not in pkl_data:
        raise KeyError("'samples' key not found in pickle")

    raw_samples = pkl_data["samples"]
    if not isinstance(raw_samples, list):
        raise TypeError(f"Expected list in pkl['samples'], got {type(raw_samples)}")

    video_meta = OrderedDict()

    for s in raw_samples:
        video_id = s["video_id"]
        img_path = s["img_path"]
        wav_path = s["wav_path"]
        fps = float(s["fps"])

        if video_id not in video_meta:
            img_dir = os.path.dirname(img_path)
            video_meta[video_id] = {
                "video_id": video_id,
                "img_dir": img_dir,
                "wav_path": wav_path,
                "fps": fps,
                "samples": [],
            }

        video_meta[video_id]["samples"].append(s)

    # Scan actual images
    for video_id, meta in video_meta.items():
        img_dir = meta["img_dir"]

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        frame_to_name = {}
        existing_frames = []

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".jpg"):
                continue
            stem = os.path.splitext(fname)[0]
            if not stem.isdigit():
                continue

            frame_idx = int(stem)
            frame_to_name[frame_idx] = fname
            existing_frames.append(frame_idx)

        existing_frames = sorted(existing_frames)

        if len(existing_frames) == 0:
            raise RuntimeError(f"No jpg images found in: {img_dir}")

        meta["frame_to_name"] = frame_to_name
        meta["existing_frames"] = existing_frames
        meta["existing_frame_set"] = set(existing_frames)

    return video_meta


# =========================================================
# 4. Test inference dataset
#    Run inference directly on all actual jpg frames
# =========================================================
class ABAWTestInferenceDataset(Dataset):
    def __init__(
        self,
        video_meta: OrderedDict,
        image_size: int = 224,
        num_image_frames: int = 20,
        window_sec: float = 10.0,
        target_sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
    ):
        super().__init__()

        self.video_meta = video_meta
        self.image_size = image_size
        self.num_image_frames = num_image_frames
        self.window_sec = window_sec
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        self.items = []
        self._build_items()

        print(f"[INFO] Number of inference items (all frames): {len(self.items)}")

    def _build_items(self):
        """
        Use all actual jpg frames as endpoints.
        """
        for video_id, meta in self.video_meta.items():
            for frame_idx in meta["existing_frames"]:
                self.items.append({
                    "video_id": video_id,
                    "target_frame": frame_idx,
                    "target_fname": meta["frame_to_name"][frame_idx],
                })

    def __len__(self):
        return len(self.items)

    def _build_sampled_image_frames(self, video_id: str, target_frame: int) -> List[int]:
        """
        Image policy:
        1) If the number of available frames so far is < 20
           -> pad with copies of the first frame to make 20 frames
        2) If there are at least 20 frames but still less than 10 seconds
           -> uniformly sample 20 frames from 1 ~ target_frame
        3) If 10 seconds or more are available
           -> uniformly sample 20 frames from the most recent 10-second window

        Note:
        Only actually existing jpg files are used.
        """
        meta = self.video_meta[video_id]
        fps = float(meta["fps"])
        existing_frames = meta["existing_frames"]

        window_num_frames = int(round(self.window_sec * fps))

        # Use only existing frames up to target_frame
        available_until_target = [f for f in existing_frames if f <= target_frame]

        if len(available_until_target) == 0:
            raise RuntimeError(f"[{video_id}] no available frames <= target_frame={target_frame}")

        first_available = available_until_target[0]

        # Case 1: fewer than 20 actual frames are available so far
        if len(available_until_target) < self.num_image_frames:
            pad_count = self.num_image_frames - len(available_until_target)
            padded = [first_available] * pad_count + available_until_target
            return padded

        # Case 2: at least 20 frames exist, but still less than 10 seconds
        if target_frame < window_num_frames:
            return sample_uniform_frames_from_list(available_until_target, self.num_image_frames)

        # Case 3: if 10 seconds or more are available, sample within the recent 10-second window
        start_frame = max(1, target_frame - window_num_frames + 1)
        window_frames = [f for f in existing_frames if start_frame <= f <= target_frame]

        if len(window_frames) == 0:
            return sample_uniform_frames_from_list(available_until_target, self.num_image_frames)

        if len(window_frames) < self.num_image_frames:
            pad_count = self.num_image_frames - len(window_frames)
            padded = [window_frames[0]] * pad_count + window_frames
            return padded

        return sample_uniform_frames_from_list(window_frames, self.num_image_frames)

    def _load_images(self, video_id: str, sampled_frames: List[int]) -> torch.Tensor:
        meta = self.video_meta[video_id]
        img_dir = meta["img_dir"]
        frame_to_name = meta["frame_to_name"]

        imgs = []
        for fidx in sampled_frames:
            if fidx not in frame_to_name:
                raise RuntimeError(f"[{video_id}] frame {fidx} not found in frame_to_name")

            img_path = os.path.join(img_dir, frame_to_name[fidx])

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            img = Image.open(img_path).convert("RGB")
            img = self.img_transform(img)
            imgs.append(img)

        return torch.stack(imgs, dim=0)  # [20, 3, H, W]

    def _load_audio_segment(self, wav_path: str, end_time_sec: float) -> torch.Tensor:
        """
        Audio policy:
        - Use only the segment available up to the current endpoint
        - Zero-pad if the segment is shorter than required
        - If longer than 10 seconds, use only the most recent 10 seconds
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        info = torchaudio.info(wav_path)
        orig_sr = info.sample_rate

        start_time_sec = max(0.0, end_time_sec - self.window_sec)

        frame_offset = int(start_time_sec * orig_sr)
        num_frames = max(1, int((end_time_sec - start_time_sec) * orig_sr))

        waveform, sr = torchaudio.load(
            wav_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )  # [channels, time]

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sample_rate,
            )

        target_len = int(round(self.window_sec * self.target_sample_rate))

        if waveform.size(1) < target_len:
            pad_amount = target_len - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_amount))
        elif waveform.size(1) > target_len:
            waveform = waveform[:, :target_len]

        return waveform  # [1, target_len]

    def _waveform_to_logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_extractor(waveform)
        mel = self.db_transform(mel)
        return mel  # [1, n_mels, time]

    def __getitem__(self, idx):
        item = self.items[idx]

        video_id = item["video_id"]
        target_frame = item["target_frame"]
        target_fname = item["target_fname"]

        fps = float(self.video_meta[video_id]["fps"])

        sampled_frames = self._build_sampled_image_frames(video_id, target_frame)
        images = self._load_images(video_id, sampled_frames)

        # Match the endpoint time calculation used in training
        end_time_sec = (target_frame - 1) / fps
        end_time_sec = max(0.0, end_time_sec)

        waveform = self._load_audio_segment(
            wav_path=self.video_meta[video_id]["wav_path"],
            end_time_sec=end_time_sec,
        )
        audio_mel = self._waveform_to_logmel(waveform)

        return {
            "video_id": video_id,
            "target_frame": target_frame,
            "target_fname": target_fname,
            "images": images,
            "audio_mel": audio_mel,
        }


def infer_collate_fn(batch):
    return {
        "video_id": [b["video_id"] for b in batch],
        "target_frame": torch.tensor([b["target_frame"] for b in batch], dtype=torch.long),
        "target_fname": [b["target_fname"] for b in batch],
        "images": torch.stack([b["images"] for b in batch], dim=0),
        "audio_mel": torch.stack([b["audio_mel"] for b in batch], dim=0),
    }


# =========================================================
# 5. Model loader
# =========================================================
def build_model_from_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    if "config" not in ckpt:
        raise KeyError("'config' not found in checkpoint")

    cfg = ckpt["config"]

    model = AVEmotionCLIPModel(
        dim=cfg.get("model_dim", 256),
        freeze_clip=cfg.get("freeze_clip", True),
        freeze_ast=cfg.get("freeze_ast", True),
        clip_model_name=cfg.get("clip_model_name", "ViT-B-16"),
        clip_pretrained=cfg.get("clip_pretrained", "openai"),
        hf_model_name=cfg.get("hf_model_name", "openai/clip-vit-base-patch16"),
        ast_model_name=cfg.get("ast_model_name", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        num_heads=cfg.get("num_heads", 4),

        # Additional settings that must be reflected from this trained model
        temporal_type=cfg.get("temporal_type", "tcn"),
        fusion_type=cfg.get("fusion_type", "cross_attn_gated"),
        tcn_levels=cfg.get("tcn_levels", 2),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        tcn_dropout=cfg.get("tcn_dropout", 0.1),
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, ckpt


# =========================================================
# 6. Inference
# =========================================================
@torch.no_grad()
def run_inference(model, loader, device: str, result_txt_path: str, clamp_output: bool = True):
    pred_rows = []

    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("image_location,valence,arousal\n")

    pbar = tqdm(loader, desc="Inference", leave=False)

    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        audio_mel = batch["audio_mel"].to(device, non_blocking=True)

        out = model(images, audio_mel)
        pred_v = out["valence"].detach().cpu().tolist()
        pred_a = out["arousal"].detach().cpu().tolist()

        batch_rows = []
        for i in range(len(pred_v)):
            video_id = batch["video_id"][i]
            target_fname = batch["target_fname"][i]

            v = float(pred_v[i])
            a = float(pred_a[i])

            if clamp_output:
                v = clamp_va(v)
                a = clamp_va(a)

            image_key = f"{video_id}/{target_fname}"
            row = (image_key, v, a)
            batch_rows.append(row)
            pred_rows.append(row)

        with open(result_txt_path, "a", encoding="utf-8") as f:
            for image_key, v, a in batch_rows:
                f.write(f"{image_key},{v:.6f},{a:.6f}\n")

    return pred_rows


# =========================================================
# 7. Validation
# =========================================================
def validate_results(video_meta: OrderedDict, final_rows: List[Tuple[str, float, float]]):
    actual_keys = []
    for video_id, meta in video_meta.items():
        for frame_idx in meta["existing_frames"]:
            fname = meta["frame_to_name"][frame_idx]
            actual_keys.append(f"{video_id}/{fname}")

    pred_keys = [row[0] for row in final_rows]

    if len(actual_keys) != len(pred_keys):
        raise AssertionError(
            f"[COUNT ERROR] actual image count={len(actual_keys)} != result row count={len(pred_keys)}"
        )

    if len(pred_keys) != len(set(pred_keys)):
        dup_counter = defaultdict(int)
        for k in pred_keys:
            dup_counter[k] += 1
        dups = [k for k, c in dup_counter.items() if c > 1][:20]
        raise AssertionError(f"[DUPLICATE ERROR] duplicated prediction keys found. examples={dups}")

    actual_set = set(actual_keys)
    pred_set = set(pred_keys)

    missing = sorted(list(actual_set - pred_set))
    extra = sorted(list(pred_set - actual_set))

    if len(missing) > 0:
        raise AssertionError(f"[MISSING ERROR] {len(missing)} missing predictions. examples={missing[:20]}")
    if len(extra) > 0:
        raise AssertionError(f"[EXTRA ERROR] {len(extra)} extra predictions. examples={extra[:20]}")

    out_of_range = []
    for image_key, v, a in final_rows:
        if not (-1.0 <= v <= 1.0) or not (-1.0 <= a <= 1.0):
            out_of_range.append((image_key, v, a))
            if len(out_of_range) >= 20:
                break

    if len(out_of_range) > 0:
        raise AssertionError(
            f"[RANGE ERROR] Found VA values outside [-1, 1]. examples={out_of_range}"
        )

    print("[VALIDATION] PASS")
    print(f"  - total images     : {len(actual_keys)}")
    print(f"  - total result rows: {len(pred_keys)}")
    print("  - duplicates       : 0")
    print("  - missing          : 0")
    print("  - extra            : 0")
    print("  - value range      : all in [-1, 1]")


# =========================================================
# 8. Main
# =========================================================
def main():
    cfg = InferConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    result_txt_path = os.path.join(cfg.output_dir, cfg.output_txt_name)

    print("[INFO] device:", cfg.device)
    print("[INFO] test_pkl:", cfg.test_pkl)
    print("[INFO] checkpoint:", cfg.checkpoint_path)

    # 1) metadata
    video_meta = build_video_metadata(cfg.test_pkl)
    print(f"[INFO] videos: {len(video_meta)}")

    total_images = sum(len(meta["existing_frames"]) for meta in video_meta.values())
    print(f"[INFO] total actual jpg images: {total_images}")

    # 2) dataset / loader
    infer_dataset = ABAWTestInferenceDataset(
        video_meta=video_meta,
        image_size=cfg.image_size,
        num_image_frames=cfg.num_image_frames,
        window_sec=cfg.window_sec,
        target_sample_rate=cfg.target_sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=infer_collate_fn,
    )

    # 3) model
    model, ckpt = build_model_from_checkpoint(cfg.checkpoint_path, cfg.device)

    if torch.cuda.device_count() >= 2 and cfg.device.startswith("cuda"):
        print(f"[INFO] Using DataParallel on GPUs: [0, 1]")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    print(f"[INFO] checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"[INFO] checkpoint best_score: {ckpt.get('best_score', 'N/A')}")

    if "config" in ckpt:
        model_cfg = ckpt["config"]
        print("[INFO] model config loaded from checkpoint:")
        print(f"       temporal_type   : {model_cfg.get('temporal_type', 'N/A')}")
        print(f"       fusion_type     : {model_cfg.get('fusion_type', 'N/A')}")
        print(f"       tcn_levels      : {model_cfg.get('tcn_levels', 'N/A')}")
        print(f"       tcn_kernel_size : {model_cfg.get('tcn_kernel_size', 'N/A')}")
        print(f"       tcn_dropout     : {model_cfg.get('tcn_dropout', 'N/A')}")

    # 4) inference
    final_rows = run_inference(
        model=model,
        loader=infer_loader,
        device=cfg.device,
        result_txt_path=result_txt_path,
        clamp_output=cfg.clamp_output,
    )

    # 5) validate
    validate_results(video_meta, final_rows)

    print(f"[INFO] result saved to: {result_txt_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
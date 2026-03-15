import os
import math
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms


# =========================================================
# 1. CLIP VA region prompt definitions
# =========================================================
def get_va_region_prompts() -> Dict[str, List[str]]:
    """
    Defines 9 VA regions x 3 templates = 27 prompts.
    These can be used in the model if needed.
    """
    region_states = [
        "sad, tired, and low-energy",                     # Low V / Low A
        "displeased and uncomfortable",                  # Low V / Mid A
        "angry, tense, and highly aroused",              # Low V / High A
        "calm, expressionless, and low-energy",          # Mid V / Low A
        "neutral and emotionally ordinary",              # Mid V / Mid A
        "alert, attentive, and emotionally neutral",     # Mid V / High A
        "relaxed, content, and pleasant",                # High V / Low A
        "happy and pleasant",                            # High V / Mid A
        "excited, joyful, and energetic",                # High V / High A
    ]

    templates = [
        "a photo of a person who looks {}",
        "a face showing {}",
        "a facial expression of {}",
    ]

    prompts = []
    region_to_prompts = []

    for state in region_states:
        one_region_prompts = [tpl.format(state) for tpl in templates]
        region_to_prompts.append(one_region_prompts)
        prompts.extend(one_region_prompts)

    return {
        "region_states": region_states,
        "templates": templates,
        "region_to_prompts": region_to_prompts,  # [9][3]
        "all_prompts": prompts,                  # [27]
    }


# =========================================================
# 2. Compute VA soft region labels
# =========================================================
def compute_soft_region_label(
    v: float,
    a: float,
    centers_1d: Tuple[float, float, float] = (-0.66, 0.0, 0.66),
    sigma: float = 0.45,
) -> torch.Tensor:
    """
    Converts a (v, a) coordinate into a soft label over 3x3 = 9 regions.
    Gaussian weighting is used.

    Returns:
        Tensor of shape [9]

    Region order:
    [LowV-LowA, LowV-MidA, LowV-HighA,
     MidV-LowA, MidV-MidA, MidV-HighA,
     HighV-LowA, HighV-MidA, HighV-HighA]
    """
    centers = []
    for v_c in centers_1d:
        for a_c in centers_1d:
            centers.append((v_c, a_c))
    # Order:
    # (-0.66,-0.66), (-0.66,0), (-0.66,0.66),
    # (0,-0.66), (0,0), (0,0.66),
    # (0.66,-0.66), (0.66,0), (0.66,0.66)

    va = np.array([v, a], dtype=np.float32)
    dists_sq = []

    for c in centers:
        c = np.array(c, dtype=np.float32)
        dist_sq = np.sum((va - c) ** 2)
        dists_sq.append(dist_sq)

    dists_sq = np.array(dists_sq, dtype=np.float32)

    # Gaussian weighting
    weights = np.exp(-dists_sq / (2 * sigma * sigma))
    weights = weights / (weights.sum() + 1e-8)

    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# 3. Dataset
# =========================================================
class ABAWAudioVisualDataset(Dataset):
    """
    Based on pkl['samples'].

    - Uses a 10-second past window ending at the current frame
    - 3-second stride
    - Uniformly samples 20 image frames
    - Uses the same 10-second window for audio log-mel extraction
    - Computes the soft region label internally

    Example returned sample:
    {
        "id": str,
        "video_id": str,
        "frame_idx": int,
        "images": Tensor [20, 3, 224, 224],
        "audio_mel": Tensor [1, n_mels, time],
        "valence": Tensor [],
        "arousal": Tensor [],
        "soft_region": Tensor [9],
        "fps": Tensor [],
        "img_frame_indices": Tensor [20],
    }
    """

    def __init__(
        self,
        pkl_path: str,
        image_size: int = 224,
        num_image_frames: int = 20,
        window_sec: float = 10.0,
        stride_sec: float = 3.0,
        centers_1d: Tuple[float, float, float] = (-0.66, 0.0, 0.66),
        sigma: float = 0.45,
        target_sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        require_endpoint_image: bool = False,
    ):
        super().__init__()

        self.pkl_path = pkl_path
        self.image_size = image_size
        self.num_image_frames = num_image_frames
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.centers_1d = centers_1d
        self.sigma = sigma

        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.require_endpoint_image = require_endpoint_image

        # -------------------------
        # Image transform
        # -------------------------
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # -------------------------
        # Audio transform
        # -------------------------
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

        # -------------------------
        # Load pickle
        # -------------------------
        with open(self.pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        if not isinstance(pkl_data, dict):
            raise TypeError(f"Expected pickle to be dict, but got {type(pkl_data)}")

        if "samples" not in pkl_data:
            raise KeyError("'samples' key not found in pickle file")

        self.raw_samples: List[Dict] = pkl_data["samples"]

        # -------------------------
        # Organize by video
        # -------------------------
        self.video_to_samples = self._group_samples_by_video(self.raw_samples)

        # Cache the list of actually existing image frame indices for each video folder
        self.video_image_info = self._scan_available_images(self.video_to_samples)

        # Build final samples after applying stride and window constraints
        self.samples = self._build_valid_samples()

        print(f"[INFO] Loaded: {self.pkl_path}")
        print(f"[INFO] Raw samples: {len(self.raw_samples)}")
        print(f"[INFO] Valid sequence samples: {len(self.samples)}")

    # -----------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------
    def _group_samples_by_video(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        video_to_samples = {}
        for s in samples:
            video_id = s["video_id"]
            video_to_samples.setdefault(video_id, []).append(s)

        for video_id in video_to_samples:
            video_to_samples[video_id] = sorted(
                video_to_samples[video_id],
                key=lambda x: x["frame_idx"]
            )
        return video_to_samples

    def _scan_available_images(self, video_to_samples: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Scans the actually existing jpg frames in each video folder.
        This explicitly handles missing frames.
        """
        video_image_info = {}

        for video_id, samples in video_to_samples.items():
            # Extract the image directory from one sample's img_path
            any_img_path = samples[0]["img_path"]
            img_dir = os.path.dirname(any_img_path)

            # Estimate filename width (e.g., 01455.jpg -> width=5)
            stem = os.path.splitext(os.path.basename(any_img_path))[0]
            filename_width = len(stem)

            existing_frames = []
            if os.path.isdir(img_dir):
                for fname in os.listdir(img_dir):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    stem = os.path.splitext(fname)[0]
                    if stem.isdigit():
                        existing_frames.append(int(stem))

            existing_frames = sorted(existing_frames)

            video_image_info[video_id] = {
                "img_dir": img_dir,
                "filename_width": filename_width,
                "existing_frames": existing_frames,
                "existing_frame_set": set(existing_frames),
            }

        return video_image_info

    def _build_valid_samples(self) -> List[Dict]:
        """
        Builds valid samples under the following rules:
        - 10-second past window ending at the current frame
        - 3-second stride
        - Excludes the first 10 seconds
        - Excludes windows with no available images
        """
        valid_samples = []

        for video_id, samples in self.video_to_samples.items():
            image_info = self.video_image_info[video_id]
            existing_frames = image_info["existing_frames"]
            existing_set = image_info["existing_frame_set"]

            if len(existing_frames) == 0:
                continue

            last_kept_time = -1e9

            for s in samples:
                fps = float(s["fps"])
                frame_idx = int(s["frame_idx"])
                end_time_sec = (frame_idx - 1) / fps  # assumes 1-based frame indexing
                start_time_sec = end_time_sec - self.window_sec

                # Do not use the initial segment shorter than 10 seconds
                if start_time_sec < 0:
                    continue

                # 3-second stride
                if (end_time_sec - last_kept_time) < self.stride_sec:
                    continue

                # Window in frame indices
                window_num_frames = int(round(self.window_sec * fps))
                start_frame = frame_idx - window_num_frames + 1
                end_frame = frame_idx

                if start_frame < 1:
                    continue

                # Optionally require the endpoint image to exist
                if self.require_endpoint_image and (frame_idx not in existing_set):
                    continue

                # Find actually existing frames within this window
                available_in_window = self._get_available_frames_in_range(
                    existing_frames, start_frame, end_frame
                )

                if len(available_in_window) == 0:
                    continue

                last_kept_time = end_time_sec

                valid_samples.append({
                    "sample": s,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_sec": start_time_sec,
                    "end_time_sec": end_time_sec,
                    "available_frames": available_in_window,
                })

        return valid_samples

    @staticmethod
    def _get_available_frames_in_range(
        sorted_existing_frames: List[int],
        start_frame: int,
        end_frame: int,
    ) -> List[int]:
        """
        Extracts only the actually existing frames within [start_frame, end_frame].
        """
        # Simple implementation
        return [f for f in sorted_existing_frames if start_frame <= f <= end_frame]

    def _sample_image_frame_indices(self, available_frames: List[int]) -> List[int]:
        """
        Uniformly samples 20 frames directly from available_frames,
        taking missing frames into account.
        If there are too few frames, duplication is allowed.
        """
        n = len(available_frames)
        if n == 0:
            raise ValueError("available_frames is empty")

        if n == 1:
            return [available_frames[0]] * self.num_image_frames

        # Uniform sampling over the available frame list
        idxs = np.linspace(0, n - 1, self.num_image_frames)
        idxs = np.round(idxs).astype(int)
        sampled = [available_frames[i] for i in idxs]
        return sampled

    def _build_image_path(self, video_id: str, frame_idx: int) -> str:
        info = self.video_image_info[video_id]
        img_dir = info["img_dir"]
        width = info["filename_width"]
        fname = f"{frame_idx:0{width}d}.jpg"
        return os.path.join(img_dir, fname)

    def _load_images(self, video_id: str, sampled_frame_indices: List[int]) -> torch.Tensor:
        imgs = []

        for frame_idx in sampled_frame_indices:
            img_path = self._build_image_path(video_id, frame_idx)

            if not os.path.exists(img_path):
                # This case should be rare, but use a black image as fallback just in case
                img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            else:
                img = Image.open(img_path).convert("RGB")

            img = self.img_transform(img)
            imgs.append(img)

        return torch.stack(imgs, dim=0)  # [T, C, H, W]

    def _load_audio_segment(self, wav_path: str, start_time_sec: float, end_time_sec: float) -> torch.Tensor:
        """
        Loads only the required 10-second segment from the wav file.

        Returns:
            Tensor of shape [1, num_samples_after_resample]
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        info = torchaudio.info(wav_path)
        orig_sr = info.sample_rate

        frame_offset = int(start_time_sec * orig_sr)
        num_frames = int((end_time_sec - start_time_sec) * orig_sr)

        waveform, sr = torchaudio.load(
            wav_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )  # [channels, time]

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sample_rate
            )

        target_len = int(round(self.window_sec * self.target_sample_rate))

        # Pad if slightly shorter than expected
        if waveform.size(1) < target_len:
            pad_amount = target_len - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_amount))
        elif waveform.size(1) > target_len:
            waveform = waveform[:, :target_len]

        return waveform  # [1, target_len]

    def _waveform_to_logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Converts waveform to log-mel spectrogram.

        Args:
            waveform: [1, time]

        Returns:
            Tensor of shape [1, n_mels, time_frames]
        """
        mel = self.mel_extractor(waveform)   # [1, n_mels, time]
        mel = self.db_transform(mel)         # log-mel
        return mel

    # -----------------------------------------------------
    # Dataset API
    # -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[index]
        s = item["sample"]

        video_id = s["video_id"]
        frame_idx = int(s["frame_idx"])
        wav_path = s["wav_path"]

        v = float(s["va"]["v"])
        a = float(s["va"]["a"])

        # -------------------------
        # Image sequence
        # -------------------------
        available_frames = item["available_frames"]
        sampled_frame_indices = self._sample_image_frame_indices(available_frames)
        images = self._load_images(video_id, sampled_frame_indices)   # [20, 3, 224, 224]

        # -------------------------
        # Audio log-mel
        # -------------------------
        waveform = self._load_audio_segment(
            wav_path=wav_path,
            start_time_sec=item["start_time_sec"],
            end_time_sec=item["end_time_sec"],
        )
        audio_mel = self._waveform_to_logmel(waveform)  # [1, n_mels, time]

        # -------------------------
        # Soft region label
        # -------------------------
        soft_region = compute_soft_region_label(
            v=v,
            a=a,
            centers_1d=self.centers_1d,
            sigma=self.sigma,
        )

        return {
            "id": s["id"],
            "video_id": video_id,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),

            "images": images,                       # [20, 3, 224, 224]
            "audio_mel": audio_mel,                # [1, n_mels, time]

            "valence": torch.tensor(v, dtype=torch.float32),
            "arousal": torch.tensor(a, dtype=torch.float32),
            "soft_region": soft_region,            # [9]

            "fps": torch.tensor(float(s["fps"]), dtype=torch.float32),
            "img_frame_indices": torch.tensor(sampled_frame_indices, dtype=torch.long),
        }
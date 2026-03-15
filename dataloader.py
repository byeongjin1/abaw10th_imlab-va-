import torch
from torch.utils.data import DataLoader

from dataset import ABAWAudioVisualDataset


def abaw_collate_fn(batch):
    """
    batch: list of dictionaries
    Groups the outputs of dataset.__getitem__ into a batch
    """

    ids = [b["id"] for b in batch]
    video_ids = [b["video_id"] for b in batch]

    frame_idx = torch.stack([b["frame_idx"] for b in batch], dim=0)         # [B]
    images = torch.stack([b["images"] for b in batch], dim=0)               # [B, 20, 3, 224, 224]
    audio_mel = torch.stack([b["audio_mel"] for b in batch], dim=0)         # [B, 1, 128, T]
    valence = torch.stack([b["valence"] for b in batch], dim=0)             # [B]
    arousal = torch.stack([b["arousal"] for b in batch], dim=0)             # [B]
    soft_region = torch.stack([b["soft_region"] for b in batch], dim=0)     # [B, 9]
    fps = torch.stack([b["fps"] for b in batch], dim=0)                     # [B]
    img_frame_indices = torch.stack([b["img_frame_indices"] for b in batch], dim=0)  # [B, 20]

    return {
        "id": ids,
        "video_id": video_ids,
        "frame_idx": frame_idx,
        "images": images,
        "audio_mel": audio_mel,
        "valence": valence,
        "arousal": arousal,
        "soft_region": soft_region,
        "fps": fps,
        "img_frame_indices": img_frame_indices,
    }


def build_datasets(
    train_pkl: str,
    val_pkl: str,
    test_pkl: str,
):
    common_kwargs = dict(
        image_size=224,
        num_image_frames=20,
        window_sec=10.0,
        stride_sec=3.0,
        centers_1d=(-0.66, 0.0, 0.66),
        sigma=0.45,
        target_sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        require_endpoint_image=False,
    )

    train_dataset = ABAWAudioVisualDataset(
        pkl_path=train_pkl,
        **common_kwargs
    )

    val_dataset = ABAWAudioVisualDataset(
        pkl_path=val_pkl,
        **common_kwargs
    )

    test_dataset = ABAWAudioVisualDataset(
        pkl_path=test_pkl,
        **common_kwargs
    )

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    train_pkl: str,
    val_pkl: str,
    test_pkl: str,
    batch_size: int = 4,
    num_workers: int = 4,
):
    train_dataset, val_dataset, test_dataset = build_datasets(
        train_pkl=train_pkl,
        val_pkl=val_pkl,
        test_pkl=test_pkl,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=abaw_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=abaw_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=abaw_collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_pkl = "/media/SSD/data/CVPR_workshop/pkl_index/train.pkl"
    val_pkl = "/media/SSD/data/CVPR_workshop/pkl_index/val.pkl"
    test_pkl = "/media/SSD/data/CVPR_workshop/pkl_index/test.pkl"

    train_loader, val_loader, test_loader = build_dataloaders(
        train_pkl=train_pkl,
        val_pkl=val_pkl,
        test_pkl=test_pkl,
        batch_size=2,
        num_workers=0,   # Recommended to test with 0 first
    )

    batch = next(iter(train_loader))

    print("ids:", batch["id"][:2])
    print("video_ids:", batch["video_id"][:2])
    print("frame_idx:", batch["frame_idx"].shape)
    print("images:", batch["images"].shape)
    print("audio_mel:", batch["audio_mel"].shape)
    print("valence:", batch["valence"].shape)
    print("arousal:", batch["arousal"].shape)
    print("soft_region:", batch["soft_region"].shape)
    print("fps:", batch["fps"].shape)
    print("img_frame_indices:", batch["img_frame_indices"].shape)
import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from dataloader import build_dataloaders
from model_clip import AVEmotionCLIPModel


# =========================================================
# 1. Config
# =========================================================
@dataclass
class TrainConfig:
    train_pkl: str = "/media/SSD/data/CVPR_workshop/pkl_index/train.pkl"
    val_pkl: str = "/media/SSD/data/CVPR_workshop/pkl_index/val.pkl"
    test_pkl: str = "/media/SSD/data/CVPR_workshop/pkl_index/test.pkl"

    batch_size: int = 16
    num_workers: int = 8

    lr_head: float = 1e-4
    lr_backbone: float = 3e-6
    weight_decay: float = 1e-4
    epochs: int = 15

    model_dim: int = 256
    lambda_region: float = 0.2

    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4  

    freeze_clip: bool = True
    freeze_ast: bool = True

    clip_model_name: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    hf_model_name: str = "openai/clip-vit-base-patch16"
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    save_dir: str = "./result/checkpoints_tcn_fusion"
    save_name: str = "best_clip_model.pt"
    log_txt: str = "train_log.txt"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    temporal_type: str = "tcn"       # "gru" or "tcn"
    # fusion_type: str = "cross_attn"        # baseline
    # fusion_type: str = "gated"           # gated only
    fusion_type: str = "cross_attn_gated" # both

    tcn_levels: int = 2
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1

# =========================================================
# 2. CCC
# =========================================================
def concordance_cc(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.view(-1)
    gt = gt.view(-1)

    mean_pred = torch.mean(pred)
    mean_gt = torch.mean(gt)

    var_pred = torch.var(pred, unbiased=False)
    var_gt = torch.var(gt, unbiased=False)

    cov = torch.mean((pred - mean_pred) * (gt - mean_gt))
    ccc = (2.0 * cov) / (var_pred + var_gt + (mean_pred - mean_gt) ** 2 + eps)
    return ccc

# # This is the plain CCC loss
class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_v, pred_a, gt_v, gt_a):
        ccc_v = concordance_cc(pred_v, gt_v)
        ccc_a = concordance_cc(pred_a, gt_a)

        loss_v = 1.0 - ccc_v
        loss_a = 1.0 - ccc_a
        loss = loss_v + loss_a

        return loss, ccc_v.detach(), ccc_a.detach()


class SoftTargetKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor):
        # logits: [B, 9], target_probs: [B, 9]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
        return loss

# =========================================================
# 3. Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, ccc_criterion, region_criterion, device, lambda_region=0.3):
    model.train()

    running_loss = 0.0
    running_ccc_v = 0.0
    running_ccc_a = 0.0
    running_region = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        audio_mel = batch["audio_mel"].to(device, non_blocking=True)
        gt_v = batch["valence"].to(device, non_blocking=True)
        gt_a = batch["arousal"].to(device, non_blocking=True)
        soft_region = batch["soft_region"].to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(images, audio_mel)
        pred_v = out["valence"]
        pred_a = out["arousal"]
        region_logits = out["region_logits"]

        ccc_loss, ccc_v, ccc_a = ccc_criterion(pred_v, pred_a, gt_v, gt_a)
        region_loss = region_criterion(region_logits, soft_region)

        loss = ccc_loss + lambda_region * region_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ccc_v += ccc_v.item()
        running_ccc_a += ccc_a.item()
        running_region += region_loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ccc_v": f"{ccc_v.item():.4f}",
            "ccc_a": f"{ccc_a.item():.4f}",
            "region": f"{region_loss.item():.4f}",
        })

    avg_loss = running_loss / max(num_batches, 1)
    avg_ccc_v = running_ccc_v / max(num_batches, 1)
    avg_ccc_a = running_ccc_a / max(num_batches, 1)
    avg_region = running_region / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "ccc_v": avg_ccc_v,
        "ccc_a": avg_ccc_a,
        "ccc_mean": (avg_ccc_v + avg_ccc_a) / 2.0,
        "region_loss": avg_region,
    }


@torch.no_grad()
def evaluate(model, loader, ccc_criterion, region_criterion, device, lambda_region=0.3, desc="Val"):
    model.eval()

    all_pred_v = []
    all_pred_a = []
    all_gt_v = []
    all_gt_a = []

    running_loss = 0.0
    running_region = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        audio_mel = batch["audio_mel"].to(device, non_blocking=True)
        gt_v = batch["valence"].to(device, non_blocking=True)
        gt_a = batch["arousal"].to(device, non_blocking=True)
        soft_region = batch["soft_region"].to(device, non_blocking=True)

        out = model(images, audio_mel)
        pred_v = out["valence"]
        pred_a = out["arousal"]
        region_logits = out["region_logits"]

        ccc_loss, _, _ = ccc_criterion(pred_v, pred_a, gt_v, gt_a)
        region_loss = region_criterion(region_logits, soft_region)
        loss = ccc_loss + lambda_region * region_loss

        running_loss += loss.item()
        running_region += region_loss.item()
        num_batches += 1

        all_pred_v.append(pred_v.detach().cpu())
        all_pred_a.append(pred_a.detach().cpu())
        all_gt_v.append(gt_v.detach().cpu())
        all_gt_a.append(gt_a.detach().cpu())

    all_pred_v = torch.cat(all_pred_v, dim=0)
    all_pred_a = torch.cat(all_pred_a, dim=0)
    all_gt_v = torch.cat(all_gt_v, dim=0)
    all_gt_a = torch.cat(all_gt_a, dim=0)

    ccc_v = concordance_cc(all_pred_v, all_gt_v).item()
    ccc_a = concordance_cc(all_pred_a, all_gt_a).item()
    avg_loss = running_loss / max(num_batches, 1)
    avg_region = running_region / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
        "ccc_mean": (ccc_v + ccc_a) / 2.0,
        "region_loss": avg_region,
    }


# =========================================================
# 4. Save / Load
# =========================================================
def save_checkpoint(save_path, model, optimizer, epoch, best_score, cfg):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "config": cfg.__dict__,
    }, save_path)


# =========================================================
# 5. Main
# =========================================================
def main():
    cfg = TrainConfig()

    print("device:", cfg.device)
    print("save_dir:", cfg.save_dir)

    # -------------------------
    # Dataloader
    # -------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        train_pkl=cfg.train_pkl,
        val_pkl=cfg.val_pkl,
        test_pkl=cfg.test_pkl,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # -------------------------
    # Model / Optimizer / Loss
    # -------------------------
    model = AVEmotionCLIPModel(
        dim=cfg.model_dim,
        freeze_clip=cfg.freeze_clip,
        freeze_ast=cfg.freeze_ast,
        clip_model_name=cfg.clip_model_name,
        clip_pretrained=cfg.clip_pretrained,
        hf_model_name=cfg.hf_model_name,
        ast_model_name=cfg.ast_model_name,
        temporal_type=cfg.temporal_type,
        fusion_type=cfg.fusion_type,
        tcn_levels=cfg.tcn_levels,
        tcn_kernel_size=cfg.tcn_kernel_size,
        tcn_dropout=cfg.tcn_dropout,
    )
    model = model.to(cfg.device)
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.startswith("image_encoder.model")
            or name.startswith("audio_encoder.ast")
        ):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    ccc_criterion = CCCLoss()
    region_criterion = SoftTargetKLLoss()

    # -------------------------
    # Train
    # -------------------------
    best_val_ccc = -1e9
    epochs_no_improve = 0
    save_path = os.path.join(cfg.save_dir, cfg.save_name)
    log_path = os.path.join(cfg.save_dir, cfg.log_txt)
    os.makedirs(cfg.save_dir, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("===== Experiment Configuration =====\n")
        f.write(f"train_pkl: {cfg.train_pkl}\n")
        f.write(f"val_pkl: {cfg.val_pkl}\n")
        f.write(f"test_pkl: {cfg.test_pkl}\n")
        f.write(f"batch_size: {cfg.batch_size}\n")
        f.write(f"num_workers: {cfg.num_workers}\n")
        f.write(f"epochs: {cfg.epochs}\n")
        f.write(f"model_dim: {cfg.model_dim}\n")
        f.write(f"lambda_region: {cfg.lambda_region}\n")
        f.write(f"lr_head: {cfg.lr_head}\n")
        f.write(f"lr_backbone: {cfg.lr_backbone}\n")
        f.write(f"weight_decay: {cfg.weight_decay}\n")
        f.write(f"freeze_clip: {cfg.freeze_clip}\n")
        f.write(f"freeze_ast: {cfg.freeze_ast}\n")
        f.write(f"clip_model_name: {cfg.clip_model_name}\n")
        f.write(f"clip_pretrained: {cfg.clip_pretrained}\n")
        f.write(f"hf_model_name: {cfg.hf_model_name}\n")
        f.write(f"ast_model_name: {cfg.ast_model_name}\n")
        f.write(f"device: {cfg.device}\n")
        f.write(f"save_dir: {cfg.save_dir}\n")
        f.write(f"save_name: {cfg.save_name}\n")
        f.write(f"temporal_type: {cfg.temporal_type}\n")
        f.write(f"fusion_type: {cfg.fusion_type}\n")
        f.write(f"tcn_levels: {cfg.tcn_levels}\n")
        f.write(f"tcn_kernel_size: {cfg.tcn_kernel_size}\n")
        f.write(f"tcn_dropout: {cfg.tcn_dropout}\n")
        f.write("\n")
        
        f.write("===== Epoch Metrics =====\n")
        f.write(
            "epoch | train_loss | train_ccc_v | train_ccc_a | train_ccc_mean | train_region | "
            "val_loss | val_ccc_v | val_ccc_a | val_ccc_mean | val_region\n"
        )

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n========== Epoch {epoch}/{cfg.epochs} ==========")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ccc_criterion=ccc_criterion,
            region_criterion=region_criterion,
            device=cfg.device,
            lambda_region=cfg.lambda_region,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            ccc_criterion=ccc_criterion,
            region_criterion=region_criterion,
            device=cfg.device,
            lambda_region=cfg.lambda_region,
            desc="Val",
        )

        print(
            f"[Train] loss={train_metrics['loss']:.4f} | "
            f"ccc_v={train_metrics['ccc_v']:.4f} | "
            f"ccc_a={train_metrics['ccc_a']:.4f} | "
            f"ccc_mean={train_metrics['ccc_mean']:.4f} | "
            f"region={train_metrics['region_loss']:.4f}"
        )

        print(
            f"[Val]   loss={val_metrics['loss']:.4f} | "
            f"ccc_v={val_metrics['ccc_v']:.4f} | "
            f"ccc_a={val_metrics['ccc_a']:.4f} | "
            f"ccc_mean={val_metrics['ccc_mean']:.4f} | "
            f"region={val_metrics['region_loss']:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch} | "
                f"{train_metrics['loss']:.4f} | "
                f"{train_metrics['ccc_v']:.4f} | "
                f"{train_metrics['ccc_a']:.4f} | "
                f"{train_metrics['ccc_mean']:.4f} | "
                f"{train_metrics['region_loss']:.4f} | "
                f"{val_metrics['loss']:.4f} | "
                f"{val_metrics['ccc_v']:.4f} | "
                f"{val_metrics['ccc_a']:.4f} | "
                f"{val_metrics['ccc_mean']:.4f} | "
                f"{val_metrics['region_loss']:.4f}\n"
            )

        if val_metrics["ccc_mean"] > best_val_ccc + cfg.early_stop_min_delta:
            best_val_ccc = val_metrics["ccc_mean"]
            epochs_no_improve = 0

            save_checkpoint(
                save_path=save_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_val_ccc,
                cfg=cfg,
            )
            print(f"[INFO] Best model saved to: {save_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            with open(log_path, "a") as f:
                f.write(f"\n[Early Stop] stopped at epoch {epoch}\n")
                f.write(f"best_val_ccc_mean: {best_val_ccc:.4f}\n")
            break
    # -------------------------
    # Test best model
    # -------------------------
    print("\n========== Final Test with Best Model ==========")
    ckpt = torch.load(save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        ccc_criterion=ccc_criterion,
        region_criterion=region_criterion,
        device=cfg.device,
        lambda_region=cfg.lambda_region,
        desc="Test",
    )

    print(
        f"[Test]  loss={test_metrics['loss']:.4f} | "
        f"ccc_v={test_metrics['ccc_v']:.4f} | "
        f"ccc_a={test_metrics['ccc_a']:.4f} | "
        f"ccc_mean={test_metrics['ccc_mean']:.4f} | "
        f"region={test_metrics['region_loss']:.4f}"
    )

    with open(log_path, "a") as f:
        f.write("\n===== Final Test Metrics =====\n")
        f.write(
            f"test_loss: {test_metrics['loss']:.4f}\n"
            f"test_ccc_v: {test_metrics['ccc_v']:.4f}\n"
            f"test_ccc_a: {test_metrics['ccc_a']:.4f}\n"
            f"test_ccc_mean: {test_metrics['ccc_mean']:.4f}\n"
            f"test_region_loss: {test_metrics['region_loss']:.4f}\n"
        )


if __name__ == "__main__":
    main()
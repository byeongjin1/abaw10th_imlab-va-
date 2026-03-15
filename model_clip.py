import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel

# ---------------------------------------------------------
# CLIP backend selection
# 1) Prefer open_clip
# 2) Fallback to transformers if unavailable
# ---------------------------------------------------------
_CLIP_BACKEND = None

try:
    import open_clip
    _CLIP_BACKEND = "open_clip"
except ImportError:
    try:
        from transformers import CLIPModel, CLIPTokenizer
        _CLIP_BACKEND = "transformers"
    except ImportError:
        _CLIP_BACKEND = None

if _CLIP_BACKEND is None:
    raise ImportError(
        "Neither open_clip nor transformers is installed.\n"
        "Install one of them:\n"
        "  pip install open_clip_torch\n"
        "or\n"
        "  pip install transformers"
    )


# =========================================================
# 1. VA region prompts
# =========================================================
def get_va_region_prompts():
    region_states = [
        "sad, tired, and low-energy",             # Low V / Low A
        "displeased and uncomfortable",           # Low V / Mid A
        "angry, tense, and highly aroused",       # Low V / High A
        "calm, expressionless, and low-energy",   # Mid V / Low A
        "neutral and emotionally ordinary",       # Mid V / Mid A
        "alert, attentive, and emotionally neutral",  # Mid V / High A
        "relaxed, content, and pleasant",         # High V / Low A
        "happy and pleasant",                     # High V / Mid A
        "excited, joyful, and energetic",         # High V / High A
    ]

    templates = [
        "a photo of a person who looks {}",
        "a face showing {}",
        "a facial expression of {}",
    ]

    region_to_prompts = []
    all_prompts = []

    for state in region_states:
        prompts = [tpl.format(state) for tpl in templates]
        region_to_prompts.append(prompts)
        all_prompts.extend(prompts)

    return {
        "region_states": region_states,
        "templates": templates,
        "region_to_prompts": region_to_prompts,  # [9][3]
        "all_prompts": all_prompts,              # [27]
    }


# =========================================================
# 2. Build fixed text prototypes for 9 VA regions
# =========================================================
def build_region_text_features(
    model_name="ViT-B-32",
    pretrained="openai",
    hf_model_name="openai/clip-vit-base-patch32",
):
    prompt_dict = get_va_region_prompts()
    region_to_prompts = prompt_dict["region_to_prompts"]
    all_prompts = prompt_dict["all_prompts"]

    if _CLIP_BACKEND == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        tokens = tokenizer(all_prompts)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)

        # open_clip text embedding dimension
        clip_dim = text_features.shape[-1]

    elif _CLIP_BACKEND == "transformers":
        model = CLIPModel.from_pretrained(hf_model_name)
        tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)

        inputs = tokenizer(
            all_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)

        clip_dim = text_features.shape[-1]

    else:
        raise ValueError("Unsupported backend")

    # 27 prompts -> 9 region prototypes
    region_features = []
    idx = 0
    for _ in range(len(region_to_prompts)):
        feats = text_features[idx:idx + 3]      # [3, clip_dim]
        proto = feats.mean(dim=0)               # [clip_dim]
        proto = F.normalize(proto, dim=-1)
        region_features.append(proto)
        idx += 3

    region_features = torch.stack(region_features, dim=0)  # [9, clip_dim]
    return region_features, clip_dim


# =========================================================
# 3. CLIP Image Encoder
#    Input: [B, T, 3, 224, 224]
#    Output:
#      proj_feat: [B, T, D]
#      clip_feat: [B, T, clip_dim]
# =========================================================
class CLIPImageEncoder(nn.Module):
    def __init__(
        self,
        out_dim=256,
        model_name="ViT-B-32",
        pretrained="openai",
        freeze_backbone=False,
        hf_model_name="openai/clip-vit-base-patch32",
    ):
        super().__init__()

        self.backend = _CLIP_BACKEND
        self.freeze_backbone = freeze_backbone

        if self.backend == "open_clip":
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
            )
            # encode_image output dimension = CLIP embedding dimension
            if hasattr(self.model.visual, "output_dim"):
                self.clip_dim = self.model.visual.output_dim
            elif hasattr(self.model, "text_projection"):
                self.clip_dim = self.model.text_projection.shape[1]
            else:
                raise ValueError("Cannot infer CLIP embedding dim for open_clip.")

        elif self.backend == "transformers":
            self.model = CLIPModel.from_pretrained(hf_model_name)
            self.clip_dim = self.model.config.projection_dim

        else:
            raise ValueError("Unsupported backend")

        self.proj = nn.Linear(self.clip_dim, out_dim)

        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def _encode_image_clip(self, x):
        if self.backend == "open_clip":
            feat = self.model.encode_image(x, normalize=False)  # [B*T, clip_dim]
            return feat

        elif self.backend == "transformers":
            feat = self.model.get_image_features(pixel_values=x)  # [B*T, clip_dim]
            return feat

        else:
            raise ValueError("Unsupported backend")

    def forward(self, x):
        """
        x: [B, T, 3, 224, 224]
        return:
            proj_feat: [B, T, out_dim]
            clip_feat: [B, T, clip_dim]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        if self.freeze_backbone:
            with torch.no_grad():
                clip_feat = self._encode_image_clip(x)
        else:
            clip_feat = self._encode_image_clip(x)

        proj_feat = self.proj(clip_feat)

        proj_feat = proj_feat.view(B, T, -1)
        clip_feat = clip_feat.view(B, T, -1)

        return proj_feat, clip_feat


# =========================================================
# 4. Audio Encoder
# =========================================================
class ASTAudioEncoder(nn.Module):
    def __init__(
        self,
        out_dim=256,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone=True,
    ):
        super().__init__()

        self.ast = ASTModel.from_pretrained(ast_model_name)
        self.ast_dim = self.ast.config.hidden_size
        self.proj = nn.Linear(self.ast_dim, out_dim)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.ast.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, 1, 128, T]
        return: [B, Ta, out_dim]
        """
        x = x.squeeze(1)  # [B, 128, T]

        # AST pretrained model expects time length 1024 by default
        if x.size(-1) < 1024:
            pad_len = 1024 - x.size(-1)
            x = F.pad(x, (0, pad_len))   # [B, 128, 1024]
        elif x.size(-1) > 1024:
            x = x[:, :, :1024]           # [B, 128, 1024]

        x = x.transpose(1, 2)            # [B, 1024, 128]

        # normalization: applied only once
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        x = x * 0.5

        if self.freeze_backbone:
            with torch.no_grad():
                out = self.ast(input_values=x)
        else:
            out = self.ast(input_values=x)

        feat = out.last_hidden_state[:, 1:, :]   # current setting: exclude only one special token
        feat = self.proj(feat)                   # [B, Ta, out_dim]
        return feat


# =========================================================
# 5. Temporal Encoder
# =========================================================
class TemporalEncoder(nn.Module):
    def __init__(self, dim=256, hidden_dim=256, num_layers=1, bidirectional=True):
        super().__init__()

        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(self.out_dim, dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.proj(out)
        return out


# =========================================================
# 5-1. TCN Block
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, T]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, dim=256, num_levels=2, kernel_size=3, dropout=0.1):
        super().__init__()

        layers = []
        in_ch = dim
        out_ch = dim

        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)          # [B, D, T]
        out = self.network(x)          # [B, D, T]
        out = out.transpose(1, 2)      # [B, T, D]
        out = self.proj(out)
        return out


# =========================================================
# 6. Cross Modal Attention
# =========================================================
class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, image_seq, audio_seq):
        attn_out, _ = self.attn(
            query=image_seq,
            key=audio_seq,
            value=audio_seq,
            need_weights=False,
        )

        x = self.norm1(image_seq + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# =========================================================
# 6-1. Gated Fusion
# =========================================================
class GatedFusion(nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()

        self.image_proj = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image_seq, audio_seq):
        """
        image_seq: [B, Ti, D]
        audio_seq: [B, Ta, D]

        Matches the output length to the image sequence length Ti
        """
        # Adaptively pool audio to match the image sequence length
        if audio_seq.size(1) != image_seq.size(1):
            audio_seq = F.adaptive_avg_pool1d(
                audio_seq.transpose(1, 2),
                output_size=image_seq.size(1)
            ).transpose(1, 2)  # [B, Ti, D]

        img = self.image_proj(image_seq)
        aud = self.audio_proj(audio_seq)

        gate = self.gate(torch.cat([img, aud], dim=-1))   # [B, Ti, D]
        fused = gate * img + (1.0 - gate) * aud
        fused = self.out_proj(fused)
        return fused


# =========================================================
# 6-2. Fusion Wrapper
#    fusion_type:
#      "cross_attn"
#      "gated"
#      "cross_attn_gated"
# =========================================================
class FusionModule(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1, fusion_type="cross_attn"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "cross_attn":
            self.cross_attn = CrossModalAttention(dim=dim, num_heads=num_heads, dropout=dropout)

        elif fusion_type == "gated":
            self.gated = GatedFusion(dim=dim, dropout=dropout)

        elif fusion_type == "cross_attn_gated":
            self.cross_attn = CrossModalAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            self.gated = GatedFusion(dim=dim, dropout=dropout)

        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, image_seq, audio_seq):
        if self.fusion_type == "cross_attn":
            fused = self.cross_attn(image_seq, audio_seq)

        elif self.fusion_type == "gated":
            fused = self.gated(image_seq, audio_seq)

        elif self.fusion_type == "cross_attn_gated":
            attended = self.cross_attn(image_seq, audio_seq)
            fused = self.gated(attended, image_seq)

        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        return fused


# =========================================================
# 7. Region Prompt Head
#    clip_feat [B, T, clip_dim] -> region_logits [B, 9]
# =========================================================
class RegionPromptHead(nn.Module):
    def __init__(self, region_text_features: torch.Tensor, logit_scale: float = 10.0):
        super().__init__()
        self.register_buffer(
            "region_text_features",
            F.normalize(region_text_features, dim=-1)
        )
        self.logit_scale = logit_scale

    def forward(self, clip_feat_seq):
        # clip_feat_seq: [B, T, clip_dim]
        clip_feat = clip_feat_seq.mean(dim=1)             # [B, clip_dim]
        clip_feat = F.normalize(clip_feat, dim=-1)

        logits = self.logit_scale * (clip_feat @ self.region_text_features.t())  # [B, 9]
        return logits


# =========================================================
# 8. VA Head
# =========================================================
class VAHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)

        out = self.mlp(x)
        valence = out[:, 0]
        arousal = out[:, 1]
        return valence, arousal


# =========================================================
# 9. Full AV Model
#    temporal_type: "gru" | "tcn"
#    fusion_type: "cross_attn" | "gated"
# =========================================================
class AVEmotionCLIPModel(nn.Module):
    def __init__(
        self,
        dim=256,
        freeze_clip=True,
        freeze_ast=True,
        clip_model_name="ViT-B-16",
        clip_pretrained="openai",
        hf_model_name="openai/clip-vit-base-patch16",
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        num_heads=4,
        temporal_type="gru",
        fusion_type="cross_attn",
        tcn_levels=2,
        tcn_kernel_size=3,
        tcn_dropout=0.1,
    ):
        super().__init__()

        self.temporal_type = temporal_type
        self.fusion_type = fusion_type

        self.image_encoder = CLIPImageEncoder(
            out_dim=dim,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            freeze_backbone=freeze_clip,
            hf_model_name=hf_model_name,
        )

        self.audio_encoder = ASTAudioEncoder(
            out_dim=dim,
            ast_model_name=ast_model_name,
            freeze_backbone=freeze_ast,
        )

        # -------------------------
        # Select temporal encoder
        # -------------------------
        if temporal_type == "gru":
            self.image_temporal = TemporalEncoder(dim=dim, hidden_dim=dim, bidirectional=True)
            self.audio_temporal = TemporalEncoder(dim=dim, hidden_dim=dim, bidirectional=True)
        elif temporal_type == "tcn":
            self.image_temporal = TCNEncoder(
                dim=dim,
                num_levels=tcn_levels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
            )
            self.audio_temporal = TCNEncoder(
                dim=dim,
                num_levels=tcn_levels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
            )
        else:
            raise ValueError(f"Unsupported temporal_type: {temporal_type}")

        # -------------------------
        # Select fusion module
        # -------------------------
        self.fusion = FusionModule(
            dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            fusion_type=fusion_type,
        )
        self.head = VAHead(dim=dim)

        region_text_features, clip_dim = build_region_text_features(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            hf_model_name=hf_model_name,
        )

        if clip_dim != self.image_encoder.clip_dim:
            raise ValueError(
                f"Text clip dim ({clip_dim}) != image clip dim ({self.image_encoder.clip_dim})"
            )

        self.region_head = RegionPromptHead(region_text_features=region_text_features)

    def forward(self, images, audio_mel):
        image_feat, image_clip_feat = self.image_encoder(images)   # [B, Ti, D], [B, Ti, clip_dim]
        audio_feat = self.audio_encoder(audio_mel)                # [B, Ta, D]

        image_feat = self.image_temporal(image_feat)
        audio_feat = self.audio_temporal(audio_feat)

        fused_feat = self.fusion(image_feat, audio_feat)

        valence, arousal = self.head(fused_feat)

        region_logits = self.region_head(image_clip_feat)

        return {
            "valence": valence,
            "arousal": arousal,
            "image_feat": image_feat,
            "audio_feat": audio_feat,
            "fused_feat": fused_feat,
            "image_clip_feat": image_clip_feat,
            "region_logits": region_logits,
        }


# =========================================================
# 10. Shape Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("clip backend:", _CLIP_BACKEND)

    model = AVEmotionCLIPModel(
        dim=256,
        freeze_clip=True,
    ).to(device)

    images = torch.randn(2, 20, 3, 224, 224).to(device)
    audio_mel = torch.randn(2, 1, 128, 313).to(device)

    with torch.no_grad():
        out = model(images, audio_mel)

    print("valence:", out["valence"].shape)
    print("arousal:", out["arousal"].shape)
    print("image_feat:", out["image_feat"].shape)
    print("audio_feat:", out["audio_feat"].shape)
    print("fused_feat:", out["fused_feat"].shape)
    print("image_clip_feat:", out["image_clip_feat"].shape)
    print("region_logits:", out["region_logits"].shape)
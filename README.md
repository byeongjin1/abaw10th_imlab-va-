
===== Environment Info =====
Python==3.12.4
Platform==Linux-5.15.0-139-generic-x86_64-with-glibc2.31
System==Linux
Release==5.15.0-139-generic
Machine==x86_64

===== PyTorch / CUDA Info =====
torch==2.4.0
CUDA available==True
CUDA version==12.4
cuDNN version==90100
GPU count==2
GPU[0]==NVIDIA GeForce RTX 3090
GPU[1]==NVIDIA GeForce RTX 3090

===== Package Versions =====
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
transformers==5.3.0
tqdm==4.66.5
Pillow==10.4.0
numpy==1.26.4
open_clip_torch==3.3.0


Our training and inference pipeline uses a metadata index file (e.g., .pkl or .csv) to describe each sample.
Each entry corresponds to a single target frame and contains the information required to load the visual frames, audio, and metadata for that sample.

Example sample entry
{
    "id": "2-30-640x360_f001001",
    "video_id": "2-30-640x360",
    "frame_idx": 1001,
    "img_path": "path/to/cropped_aligned_image/2-30-640x360/01001.jpg",
    "wav_path": "path/to/audio/2-30-640x360.wav",
    "video_path": "path/to/video/2-30-640x360.mp4",
    "fps": 30.0,
    "va": {
        "v": ...,
        "a": ...
    },
    "meta": {
        "ann_path": "path/to/annotation_or_split_file.txt",
        "base_video_id": "2-30-640x360"
    }
}

How to Run
1. Prepare the metadata files

Before training or inference, prepare the metadata index files (e.g., .pkl or .csv) that describe the dataset samples.

Example fields are described in the Data Format section above.

2. Train the model

To train the audio-visual valence-arousal model, run:

python train.py

This script:

loads the train / validation / test metadata files

builds the dataloaders

trains the model using CCC loss and region-based auxiliary loss

saves the best checkpoint based on validation CCC

Main training options are defined in TrainConfig inside train.py.

3. Run inference on the test set

To generate frame-level predictions for the test set, run:

python inference_test.py

This script:

loads the trained checkpoint

runs inference on all available test frames

saves predictions in .txt format

The output file is saved as:

result.txt

or according to the path specified in InferConfig.

4. Fill missing predictions to match the official format

If the official submission format contains image entries that are missing from the generated predictions, run:

filled_undetectable_image.ipynb

This script:

reads the official format file

aligns the prediction order to the required submission format

fills missing entries using the nearest available previous or next prediction

Main Configuration Options

The model supports different temporal encoders and fusion strategies.

Temporal encoder

Defined by:

temporal_type = "tcn"   # "gru" or "tcn"

gru: uses a bidirectional GRU as the temporal encoder

tcn: uses a Temporal Convolutional Network (TCN) as the temporal encoder

In our experiments, this option controls how temporal information is modeled from the image and audio sequences.

Fusion strategy

Defined by:

# fusion_type = "cross_attn"        # baseline
# fusion_type = "gated"             # gated only
fusion_type = "cross_attn_gated"    # both

cross_attn: uses cross-modal attention to fuse image and audio features

gated: uses gated fusion only

cross_attn_gated: first applies cross-modal attention, then applies gated fusion

This option determines how visual and audio features are combined before the final VA prediction head.

Example workflow

A typical workflow is:

python train.py
python inference_test.py
filled_undetectable_image.ipynb


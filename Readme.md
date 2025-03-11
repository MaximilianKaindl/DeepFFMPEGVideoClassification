# DeepFFmpeg Video Classification

![DeepFFmpeg Banner](https://img.shields.io/badge/DeepFFmpeg-AI%20Video%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Latest-red)

This Project demonstrates a simplified interface for interaction with the new FFmpeg DNN classification filter that uses CLIP and CLAP. This implementation currently is only availiable on my Fork.

## 🔍 Features

- **Multi-modal Analysis**: CLIP visual understanding, and CLAP audio classification for comprehensive media analysis. (Sample Detection Model and Command Builder also included)
- **GPU Acceleration**: Leverage CUDA for faster processing
- **Model Conversion**: Automated tools to convert and test CLIP and CLAP models

## 📋 Requirements

- Python 3.10.16
- FFmpeg (Submodule included, clone recursively)
- LibTorch C++ libraries (Download as described below)
- tokenizers-cpp (Submodule included, clone recursively)
- OpenVINO Toolkit (optional, for detection only)
- GPU support (optional but recommended)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/MaximilianKaindl/DeepFFMPEGVideoClassification.git
cd DeepFFMPEGVideoClassification
```

### 2. Install Cuda FFmpeg dependencies (Optional, not needed for basic CLIP/CLAP or Detection)

```bash
# needed for CUDA accel
sudo apt install nvida-cuda-toolkit
```

### 3. Build tokenizers-cpp

```bash
# Clone the repository with submodules if not already existing
git clone --recurse-submodules https://github.com/mlc-ai/tokenizers-cpp.git
cd tokenizers-cpp/example/
./build_and_run.sh
```

### 4. Download and extract LibTorch

```bash
# Download and extract LibTorch (C++ libraries) from https://pytorch.org/get-started/locally/

# sample installation Linux Libtorch CPU
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.6.0%2Bcpu.zip

# sample installation Linux Libtorch CUDA 12.6
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126.zip -d /path/to/install
```

### 5. (Optional) Download OpenVINO Toolkit

```bash
# Download OpenVINO Toolkit
https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux     # choose the appropriate version
tar -xzf openvino_2025(....) -C /path/to/install
```

### 6. Configure and build FFmpeg

```bash
# Clone the FFmpeg fork
git clone https://github.com/MaximilianKaindl/FFmpeg.git
cd FFmpeg

# take a look at what parts you need, use --cuda if downloaded Libtorch with cuda
./setup.sh --help

# Important - Set installations paths of libtorch, tokenizers-cpp and openvino
# Lines 48-50 in the script
# default setting is
# ./libtorch
# ./tokenizers-cpp
# /opt/intel/openvino

# Set up environment 
source ./setup.sh   # starts FFmpeg configure

# Clean previous builds
make clean

# Build FFmpeg
make -j16
```

### 7. Set bashrc (optional)

```bash
#env variables are only saved in current terminal so consider adding the env variables to ~/.bashrc
./setup.sh --print-bashrc
```

### 8. Install Python env

#### Python Environment Setup

This project can be set up using either Conda or Python's built-in venv module. Choose the method that best fits your workflow.

##### Option 1: Using Conda

###### Installation Steps

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) if you don't have it already.

2. Create and activate a new conda environment:

```bash
# Create environment from environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate deepffmpegvideoclassification
```

##### Option 2: Using Python venv

###### Installation Steps

1. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Model Conversion

The project includes a model conversion tool (`run_conversion.py`) that handles downloading, converting, and testing CLIP and CLAP models:

```bash
# Convert both CLIP and CLAP models
python run_conversion.py

# Convert only CLIP model
python run_conversion.py --skip-clap

# Convert only CLAP model
python run_conversion.py --skip-clip

# Use GPU acceleration during conversion
python run_conversion.py --use-cuda

# Usage Prompt
python run_conversion.py --usage
```

## 🔧 Usage

### Running a basic visual analysis with CLIP

If Model testing was done after conversion, a models_config.json gets created in /models. 
If not specified in the arguments, those models and tokenizers will be default.

```bash
python run_ffmpeg.py \
  --input resources/images/cat.jpg \
  --clip-labels resources/labels/labels_clip_animal.txt

# Hint: default temperature is high, so results are not that decisive 

python run_ffmpeg.py \
  --input resources/video/example.mp4 \
  --scene-threshold 0.4 \
  --clip-categories resources/labels/categories_clip.txt \

# Usage Prompt
python run_ffmpeg.py --usage
```

### Running audio analysis with CLAP

```bash
python run_ffmpeg.py \
  --input resources/audio/blues.mp3 \
  --clap-labels resources/labels/labels_clap_music.txt \
```

### Complete pipeline with detection and CLIP
only works when FFMPEG is built with Openvino
```bash
python run_ffmpeg.py \
  --input resources/video/sample.mp4 \
  --scene-threshold 0.4 \
  --confidence 0.1 \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml \
  --detect-labels resources/labels/coco_80cl.txt \
  --anchors "81&82&135&169&344&319" \
  --nb-classes 80 \
  --clip-labels resources/labels/labels_clip_person.txt \
```

## 📊 Output Format

The analysis results are saved in a structured csv format:

```
stream_id,label,avg_probability,count
0,Home Recording,0.5168,11
0,Narrative Film,0.3436,2
0,Action,0.2122,1
0,Horror/Thriller,0.2254,10
0,Documentary,0.3437,8
0,Non-Narrative Media,0.3423,1
```

## 📁 Project Structure

```
deepffmpeg/
├── converters/                  # Model conversion scripts
│   ├── clip_to_pt.py            # CLIP model converter
│   ├── clap_to_pt.py            # CLAP model converter
│   └── test_scripted_models.py  # Testing tools
├── models/                      # Directory for model storage
│   ├── clip/                    # CLIP models
│   ├── clap/                    # CLAP models
│   └── detect/                  # Detection models
├── resources/                   # Resource files
│   ├── audio/                   # Sample audio files
│   ├── images/                  # Sample images
│   ├── labels/                  # Label files
│   └── video/                   # Sample videos
├── environment.yml              # Conda environment file
├── requirements.txt             # Pip requirements file
├── run_conversion.py            # Model conversion script
├── run_ffmpeg.py                # Main script for running analysis
```

## ⚠️ Known Issues and Troubleshooting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [CLAP: Learning Audio Concepts from Natural Language Supervision](https://github.com/microsoft/CLAP)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [OpenVINO Toolkit](https://docs.openvino.ai/)

> **Note:** Some of these references may be missing or have moved since this documentation was created. Please check for the most up-to-date sources if links are broken.

## 🙏 Acknowledgements

- The OpenAI CLIP team for their groundbreaking work
- Microsoft Research for the CLAP model
- The FFmpeg community for their continuous development of the platform
- The PyTorch team for LibTorch

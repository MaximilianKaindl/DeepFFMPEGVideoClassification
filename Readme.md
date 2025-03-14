# DeepFFmpeg Video Classification

![DeepFFmpeg Banner](https://img.shields.io/badge/DeepFFmpeg-AI%20Video%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Latest-red)

## Table of Contents
- [Overview](#overview)
- [Core Technology](#core-technology)
- [Key Features](#key-features)
- [Practical Applications](#practical-applications)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Optional CUDA Dependencies](#2-install-optional-cuda-dependencies)
  - [3. Build Tokenizers](#3-build-tokenizers)
  - [4. Install LibTorch](#4-install-libtorch)
  - [5. Install OpenVINO (Optional)](#5-install-openvino-optional)
  - [6. Configure and Build FFmpeg](#6-configure-and-build-ffmpeg)
  - [7. Set Environment Variables (Optional)](#7-set-environment-variables-optional)
  - [8. Set Up Python Environment](#8-set-up-python-environment)
- [Model Conversion](#model-conversion)
- [Usage](#usage)
  - [Visual Analysis with CLIP](#visual-analysis-with-clip)
  - [Audio Analysis with CLAP](#audio-analysis-with-clap)
  - [CLIP and CLAP Analysis](#clip-and-clap-analysis)
  - [Pipeline with Detection](#pipeline-with-detection)
  - [Technical Media Analysis](#technical-media-analysis)
  - [Combined Media Analysis](#combined-media-analysis)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Overview
DeepFFmpeg integrates modern zero-shot classification AI models directly into the FFmpeg processing pipeline, offering sophisticated content analysis without requiring additional workflows. This practical approach makes semantic media understanding accessible through standard FFmpeg commands, providing an efficient solution for developers and media professionals.

## Core Technology

At its heart, DeepFFmpeg leverages two powerful zero-shot learning models:

1. **CLIP (Contrastive Language-Image Pre-training)** - Provides deep visual understanding by connecting image content with natural language descriptions
2. **CLAP (Contrastive Language-Audio Pre-training)** - Delivers semantic audio analysis through similar language-based classification

The framework embeds these models as native FFmpeg filters, allowing for real-time processing and analysis of media content with minimal overhead.

## Key Features

- **Unified Processing Pipeline**: Perform AI-powered analysis within your existing FFmpeg workflows
- **Multi-modal Understanding**: Analyze both visual and audio components simultaneously for comprehensive media insights
- **Zero-shot Learning**: Classify content using natural language prompts without requiring specific training data
- **Technical Metadata Integration**: Combine AI classification with detailed technical media analysis
- **Flexible Output Options**: Generate structured JSON reports, CSV classifications, or visualization overlays
- **GPU Acceleration**: Leverage CUDA for faster processing when available
- **Advanced Scene Analysis**: Identify scene changes, emotional moments, and content structure

## Practical Applications

DeepFFmpeg enables developers and media professionals to:

- Automatically categorize and tag large media libraries
- Extract semantic insights from video and audio content
- Identify content type, quality, mood, and narrative structure
- Build intelligent media applications with deep content understanding

The project provides a complete ecosystem including model conversion tools, command builders, and analysis utilities, making it accessible for both technical and creative applications in media processing.

## Requirements

- Python 3.11
- FFmpeg (Submodule included, clone recursively)
- LibTorch C++ libraries
- tokenizers-cpp library (Submodule included)
- OpenVINO Toolkit (optional, for detection only)
- GPU support (optional but recommended)

## Linux Installation

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/MaximilianKaindl/DeepFFMPEGVideoClassification.git
cd DeepFFMPEGVideoClassification
```

### 2. Install Optional CUDA Dependencies

```bash
# Only needed for CUDA acceleration
sudo apt install nvidia-cuda-toolkit
```

### 3. Build Tokenizers

```bash
# install rust and reset the terminal
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
reset

# install following dependencies
sudo apt install unzip
sudo apt install make
sudo apt install cmake
sudo apt-get install build-essential
sudo apt-get update

cd tokenizers-cpp/example/
# enshure all submodules have all files
git submodule update --init --recursive
./build_and_run.sh
```

### 4. Install LibTorch

Download and extract LibTorch C++ libraries from the [PyTorch website](https://pytorch.org/get-started/locally/):

```bash
# Installation recommended in Project directory for ease of setup
cd ../../
# Should end up in Project Directory if installation was followed

# CPU version (Linux)[
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip # Linux C++ CPU Debug
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cpu.zip -d ./

# CUDA 12.6 version (Linux)
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip # Linux C++ CUDA 12.6 Debug
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126.zip -d ./
```

### 5. Install OpenVINO (Optional)

Only required for object detection functionality:

```bash
# Download OpenVINO Toolkit 2023.3 LTS recommended
https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux
tar -xzf /path/of/download -C /path/to/install

# Set Path accordingly  
export PKG_CONFIG_PATH=/opt/intel/openvino/runtime/lib/intel64/pkgconfig:$PKG_CONFIG_PATH
# initialize 
source /opt/intel/openvino/setupvars.sh

# Example Installation for Ubuntu 22
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64.tgz
sudo mkdir /opt/intel
sudo tar -xzf l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64.tgz -C /opt/intel
# rename
sudo mv /opt/intel/l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64 /opt/intel/openvino_2023.3
# Create symbolic link
sudo ln -s /opt/intel/openvino_2023.3 /opt/intel/openvino
```

### 6. Configure and Build FFmpeg

```bash
cd FFmpeg

# View available options
./setup.sh --help

# Important: Edit setup.sh to set installation paths (lines 73-75)
# Default paths are:
# ./libtorch
# ./tokenizers-cpp
# /opt/intel/openvino

# Set up environment and configure FFmpeg
source ./setup.sh
# If terminal crashes - setup failed, run without source to see error

# Clean previous builds
make clean

# Build FFmpeg
make -j$(nproc)
```

### 7. Set Environment Variables (Optional)

To persist environment variables across terminal sessions:

```bash
./setup.sh --print-bashrc
# Add the printed variables to your ~/.bashrc file
```

### 8. Set Up Python Environment

Choose one of the following methods:

#### Using Conda

```bash
# Create environment from environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate deepffmpegvideoclassification
```

#### Using Python venv

```bash
Copy# First install software-properties-common which provides add-apt-repository
sudo apt install software-properties-common

# Now you can add the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11

# Install pip for Python 3.11
sudo apt install python3.11-venv python3.11-dev python3-pip

# create alias
alias python=python3.11

# Create a virtual environment with Python 3.11 specifically
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Model Conversion

The project includes a model conversion tool for downloading, converting, and testing CLIP and CLAP models:

```bash
# Convert both CLIP and CLAP models
python run_conversion.py

# Convert only CLIP model
python run_conversion.py --skip-clap

# Convert only CLAP model
python run_conversion.py --skip-clip

# Use GPU acceleration during conversion
python run_conversion.py --use-cuda

# View all options
python run_conversion.py --usage
```

## Usage

### Visual Analysis with CLIP

After conversion, a `models_config.json` file is created in the `/models` directory with default model paths.

```bash
# Basic image classification
python run_classification.py \
  --input resources/images/cat.jpg \
  --device cuda \
  --temperature 0.07 \
  --clip-labels resources/labels/labels_clip_animal.txt

# Video scene analysis
python run_classification.py \
  --input resources/video/example.mp4 \
  --device cuda \
  --scene-threshold 0.4 \
  --clip-categories resources/labels/categories_clip.txt 

# View all options
python run_classification.py --usage
```

### Audio Analysis with CLAP

```bash
# Audio classification 
python run_classification.py \
  --input resources/audio/blues.mp3 \
  --clap-labels resources/labels/labels_clap_music.txt 
```

### CLIP and CLAP Analysis

```bash
python run_classification.py \
--input resources/video/example.mp4 \
--device cuda \
--clip-categories resources/labels/categories_clip.txt \
--clap-categories resources/labels/categories_clap.txt \
--scene-threshold 0.2 \
--temperature 0.1
```

### Pipeline with Detection

Only works when FFmpeg is built with OpenVINO support.

For YOLO model download and conversion pleaser refer to the `converters/convert_yolo.sh`.

```bash
python run_classification.py \
  --input resources/video/sample.mp4 \
  --device cuda \
  --scene-threshold 0.4 \
  --confidence 0.1 \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml \
  --detect-labels resources/labels/coco_80cl.txt \
  --anchors "81&82&135&169&344&319" \
  --nb-classes 80 \
  --clip-labels resources/labels/labels_clip_person.txt 

# FFplay - play video with drawn detection bbox and classification label
# If FFplay is not working check your setup configuration (enable --draw)
# very low temperature so model decides on a classification

python run_classification.py \
  --input resources/video/johnwick.mp4 \
  --device cuda \
  --visualization \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml \
  --detect-labels resources/labels/coco_80cl.txt \
  --confidence 0.4 \
  --clip-labels resources/labels/labels_clip_person.txt \
  --temperature 0.007 \
  --box-color red \
  --text-color yellow

```

### Technical Media Analysis

New functionality for extracting detailed technical metadata and analyzing media structure:

```bash
# Basic technical analysis
python run_analysis.py input_video.mp4 

# Specify output directory and custom scene threshold
python run_analysis.py input_video.mp4 \
  -o analysis_results \
  --scene-threshold 0.2

# Save analysis results to specific JSON file
python run_analysis.py input_video.mp4 \
  --json results/detailed_analysis.json

# Verbose output for troubleshooting
python run_analysis.py input_video.mp4 -v
```

The technical analysis provides:
- Detailed metadata (codec, resolution, frame rate, bit depth, etc.)
- Scene detection and analysis (timestamps, average scene duration)
- Audio analysis (volume levels, silence/action moments)
- System information (recording device clues, quality classification)

### Combined Media Analysis

Integrates technical analysis with AI classification to provide comprehensive insights:

```bash
# Run combined analysis with default settings
python run_combined.py input_video.mp4 --device cuda -o combined_results

# Use existing analysis files (skip reanalysis and classification)
python run_combined.py input_video.mp4 \
  --from-existing path/to/technical_analysis.json \
  --classification-txt path/to/classifications.txt
```

The combined analysis uses the `resources/labels/clip_combined_analysis.txt` and `resources/labels/clap_combined_analysis.txt`. The Categories names must be constant, dont change them.

#### How Combined Analysis Works

Think of the combined analysis as connecting two different ways of understanding a video:

1. **Technical Analysis**: Examines "how" the video was made - its resolution, scene changes, audio levels, etc.
2. **AI Classification**: Recognizes "what" is in the video - both visually and through sound.

The system uses simple rules to combine these inputs and generate meaningful insights about content type, quality, mood, and storytelling approach - similar to how a human media analyst would evaluate content, but in an automated way.

#### Content Type Classification

The system classifies videos into three primary categories:

- **Storytelling**: Narrative-driven content with a beginning, middle, and end
- **Informational**: Content that focuses on conveying facts or information
- **Entertainment**: Content designed primarily for amusement or engagement

Classification is based on:
- Visual and audio classification results
- Scene change patterns
- Duration of shots
- Audio characteristics (speech vs. music ratio)

#### Quality Assessment
- **Video Quality**: Based on resolution, bitrate, encoding parameters, and consistency
- **Audio Quality**: Based on sample rate, bit depth, noise levels, and clarity
- **Production Value**: Estimated based on technical markers of professional production

#### Mood Analysis
- **Primary Mood**: The dominant emotional tone of the content
- **Mood Progression**: How the mood changes throughout the video
- **Mood Consistency**: Whether the mood remains stable or varies significantly

#### Storytelling Metrics
For content identified as narrative-driven:
- **Narrative Structure**: Identification of classic story structures
- **Pacing**: Rhythm and speed of content delivery
- **Key Moments**: Detection of potential climactic or emotionally significant scenes

## Output Format

The following example was created by the following command and the default models from the conversion.
Analysis results are saved in various formats depending on the analysis type:

```bash
python run_combined.py --scene-threshold 0.1 --device cuda -o combined_analysis resources/video/Popeye.mp4

# the combined script just runs those commands with added default parameters
python run_classification.py \
  --input resources/video/Popeye.mp4 \
  --scene-threshold 0.1 \
  --temperature 0.1 \
  --clip-categories resources/labels/clip_combined_analysis.txt \
  --clap-categories resources/labels/clap_combined_analysis.txt \
  --output-stats combined_analysis/Popeye_classifications.txt \
  --skip-confirmation \
  --device cuda 

python  run_analysis.py resources/video/Popeye.mp4 \
  --scene-threshold 0.1 \
  --json combined_analysis/Popeye_technical.json 
```



### AI Classification
When running visual or audio AI classification, results are saved in a structured CSV format.

View a sample result in `examples/clip_clap.csv`.

- `stream_id`: 0 for video stream, 1 for audio stream
- `label`: The classification label from the specified categories file
- `avg_probability`: Average probability/confidence score for this label
- `count`: Number of frames/segments with this label as top classification

### Technical Analysis Output

Technical analysis generates a detailed JSON file with hierarchical information: `examples/technical_analysis.json`

### Combined Analysis Output

The combined analysis generates the most comprehensive JSON output: examples/combined_analysis.json

##### Command Line Summary Output

In addition to saving files, the tools also print a human-readable summary to the console:

```
=== Combined Media Analysis Summary ===
File: resources/video/Popeye.mp4
Size: 101.27 MB
Duration: 1019.32 seconds

Content Type: Storytelling (16.67% confidence)
Subtypes: Animated, Exciting, EerieAudio

Video Quality: Standard
Audio Quality: Basic

Technical Highlights:
- Video: 720x480 @ 29.97fps (h264)
- Audio: aac 44100Hz 2ch
- Scenes: 203 (avg 5.02s)

Mood: Eerie (70.80% confidence)
Mood Consistency: Moderately consistent

Narrative Structure: Rising Action
Pacing: Fast

Key Moments:
- 7.93s: Major action/emotional moment (intensity: 1.0)
- 65.93s: Major action/emotional moment (intensity: 1.0)
- 127.04s: Major action/emotional moment (intensity: 1.0)

Top Video Classifications:
- Animation: 0.79 (count: 108)
- LowQuality: 0.70 (count: 106)
- Exciting: 0.50 (count: 101)

Top Audio Classifications:
- EerieAudio: 1.00 (count: 141)
- BasicRecording: 0.98 (count: 113)
- InformationalAudio: 0.95 (count: 77)
```

This multi-level output format allows you to access both raw analysis data and derived insights, making it suitable for both technical and creative applications.

## Project Structure

```
deepffmpeg/
├── converters/                  # Model conversion scripts
│   ├── clip_to_pt.py            # CLIP model converter
│   ├── clap_to_pt.py            # CLAP model converter
│   ├── convert_yolo.sh          # YOLO conversion script
│   └── test_scripted_models.py  # Testing tools
├── models/                      # Directory for model storage
│   ├── clip/                    # CLIP models
│   ├── clap/                    # CLAP models
│   ├── detect/                  # Detection models
│   └── models_config.json       # Model configuration file
├── resources/                   # Resource files
│   ├── audio/                   # Sample audio files
│   ├── images/                  # Sample images
│   ├── labels/                  # Label files for classification
│   │   ├── categories_clap.txt  # Sample CLAP category labels
│   │   ├── categories_clip.txt  # Sample CLIP category labels 
│   │   ├── clap_combined_analysis.txt  # Audio labels for combined analysis
│   │   ├── clip_combined_analysis.txt  # Visual labels for combined analysis
│   │   ├── coco_80cl.txt        # COCO dataset labels for detection
│   │   ├── labels_clap_music.txt # CLAP music genre labels
│   │   └── labels_clip_person.txt # CLIP person activity labels
│   └── video/                   # Sample videos
├── FFmpeg/                      # FFmpeg submodule (custom fork)
| └──── setup.sh                 # configuration script
├── tokenizers-cpp/              # Tokenizers C++ library submodule
├── environment.yml              # Conda environment file
├── requirements.txt             # Pip requirements file
├── run_conversion.py            # Model conversion script
├── run_classification.py        # Video/audio classification script
├── run_analysis.py              # Technical media analysis script
├── run_combined.py              # Combined analysis script
```

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are properly installed
2. Check environment variables are correctly set
3. For CUDA issues, verify NVIDIA drivers and CUDA toolkit are compatible
4. For model conversion problems, check Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [CLAP: Learning Audio Concepts from Natural Language Supervision](https://github.com/microsoft/CLAP)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [OpenVINO Toolkit](https://docs.openvino.ai/)

## Acknowledgements

- The OpenAI CLIP team for their groundbreaking work
- Microsoft Research for the CLAP model
- The FFmpeg community for their continuous development
- The PyTorch team for LibTorch
- The mlc-ai Team for their cross-platform C++ tokenizer binding library

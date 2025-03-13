# DeepFFmpeg Video Classification

![DeepFFmpeg Banner](https://img.shields.io/badge/DeepFFmpeg-AI%20Video%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Latest-red)

A simplified interface for using FFmpeg's DNN classification filter with CLIP and CLAP models. This implementation is currently available only on the author's fork, which is included as a submodule.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
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

DeepFFmpeg Video Classification provides a framework for advanced media content analysis using deep neural networks. It integrates CLIP (Contrastive Language-Image Pre-training) for visual understanding and CLAP (Contrastive Language-Audio Pre-training) for audio classification into FFmpeg's filtering system.

## Features

- **Multi-modal Analysis**: Combine CLIP visual understanding and CLAP audio classification for comprehensive media analysis
- **GPU Acceleration**: Leverage CUDA for faster processing
- **Command Builder**: Simplified interface for building complex FFmpeg commands
- **Model Conversion**: Automated tools to convert and test CLIP and CLAP models
- **Technical Metadata Extraction**: Detailed analysis of media file technical properties
- **Advanced Scene Detection**: Identify scene changes and key moments
- **Audio Analysis**: Volume levels, silence detection, and action moments
- **Combined Insights**: Derive content type, quality assessment, mood, and storytelling metrics

## Requirements

- Python 3.10.16
- FFmpeg (Submodule included, clone recursively)
- LibTorch C++ libraries
- tokenizers-cpp library (Submodule included)
- OpenVINO Toolkit (optional, for detection only)
- GPU support (optional but recommended)

## Installation

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
# Build tokenizers-cpp if not already existing
git clone --recurse-submodules https://github.com/mlc-ai/tokenizers-cpp.git
cd tokenizers-cpp/example/
./build_and_run.sh
```

### 4. Install LibTorch

Download and extract LibTorch C++ libraries from the [PyTorch website](https://pytorch.org/get-started/locally/):

```bash
# CPU version (Linux)
wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.6.0%2Bcpu.zip
unzip libtorch-win-shared-with-deps-2.6.0+cpu.zip -d /path/to/install

# CUDA 12.6 version (Linux)
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126.zip -d /path/to/install
```

### 5. Install OpenVINO (Optional)

Only required for object detection functionality:

```bash
# Download OpenVINO Toolkit
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux/openvino_2025.0.0.tgz
tar -xzf openvino_2025.0.0.tgz -C /path/to/install
```

### 6. Configure and Build FFmpeg

```bash
# Clone the FFmpeg fork
git clone https://github.com/MaximilianKaindl/FFmpeg.git
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
# Create a virtual environment
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
  --clip-labels resources/labels/labels_clip_animal.txt

# Video scene analysis
python run_classification.py \
  --input resources/video/example.mp4 \
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
  --scene-threshold 0.2 \
  --input resources/video/example.mp4 \
  --temperature 0.1 \ 
  --clip-categories resources/labels/categories_clip.txt \ 
  --clap-categories resources/labels/categories_clap.txt \
```

### Pipeline with Detection

Only works when FFmpeg is built with OpenVINO support.

For YOLO model download and conversion pleaser refer to the `converters/convert_yolo.sh`.

```bash
python run_classification.py \
  --input resources/video/sample.mp4 \
  --scene-threshold 0.4 \
  --confidence 0.1 \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml \
  --detect-labels resources/labels/coco_80cl.txt \
  --anchors "81&82&135&169&344&319" \
  --nb-classes 80 \
  --clip-labels resources/labels/labels_clip_person.txt 

# FFplay - play video with drawn detection bbox and classification label
# If FFplay is not working check your setup configuration
# very low temperature so model decides on a classification

python run_classification.py \
  --input resources/video/johnwick.mp4 \
  --visualization \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml    \
  --detect-labels resources/labels/coco_80cl.txt   \
  --confidence 0.4 --clip-labels resources/labels/labels_clip_person.txt \
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
  --threshold 0.2

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
python run_combined.py input_video.mp4

# Specify custom categories and output path
python run_combined.py input_video.mp4 \
  -o combined_results \
  --json results/combined_insights.json

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

Quality metrics include:

- **Video Quality**: Based on resolution, bitrate, encoding parameters, and consistency
- **Audio Quality**: Based on sample rate, bit depth, noise levels, and clarity
- **Production Value**: Estimated based on technical markers of professional production

#### Mood Analysis

The mood analysis provides:

- **Primary Mood**: The dominant emotional tone of the content
- **Mood Progression**: How the mood changes throughout the video
- **Mood Consistency**: Whether the mood remains stable or varies significantly

#### Storytelling Metrics

For content identified as narrative-driven:

- **Narrative Structure**: Identification of classic story structures
- **Pacing**: Rhythm and speed of content delivery
- **Key Moments**: Detection of potential climactic or emotionally significant scenes

## Output Format

Analysis results are saved in a structured CSV format:

```
stream_id,label,avg_probability,count
0,Home Recording,0.5168,11
0,Narrative Film,0.3436,2
0,Action,0.2122,1
0,Horror/Thriller,0.2254,10
0,Documentary,0.3437,8
0,Non-Narrative Media,0.3423,1
```

Technical and combined analyses are saved as JSON files with detailed hierarchical information about the media file.

## Project Structure

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
├── run_classification.py        # Main script for running analysis
├── run_analysis.py              # Technical media analysis script
├── run_combined.py              # Combined analysis (technical + AI classification)
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

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

Analysis results are saved in various formats depending on the analysis type:

### Classification Output

When running visual or audio classification, results are saved in a structured CSV format:

```
stream_id,label,avg_probability,count
0,Home Recording,0.5168,11
0,Narrative Film,0.3436,2
0,Action,0.2122,1
0,Horror/Thriller,0.2254,10
0,Documentary,0.3437,8
0,Non-Narrative Media,0.3423,1
1,Spoken Word,0.4532,15
1,Music,0.3241,7
1,Professional Recording,0.5123,12
```

- `stream_id`: 0 for video stream, 1 for audio stream
- `label`: The classification label from the specified categories file
- `avg_probability`: Average probability/confidence score for this label
- `count`: Number of frames/segments with this label as top classification

### Technical Analysis Output

Technical analysis generates a detailed JSON file with hierarchical information:

```json
{
  "input_file": "path/to/input_video.mp4",
  "metadata": {
    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
    "duration": 185.42,
    "size_bytes": 31457280,
    "bitrate": 2048000,
    "video_streams": [
      {
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "fps": 29.97,
        "profile": "High",
        "bit_depth": 8,
        "color_space": "bt709"
      }
    ],
    "audio_streams": [
      {
        "codec": "aac",
        "sample_rate": 48000,
        "channels": 2,
        "channel_layout": "stereo"
      }
    ]
  },
  "scene_analysis": {
    "scene_count": 42,
    "average_scene_duration": 4.41,
    "scene_timestamps": [0.0, 4.41, 8.82, ...]
  },
  "audio_analysis": {
    "mean_volume": -18.3,
    "max_volume": -9.7,
    "silence_moments": [
      {"time": 23.5, "duration": 2.1}
    ],
    "action_moments": [
      {"time": 45.2, "type": "volume_peak", "intensity": 0.85},
      {"time": 92.7, "type": "volume_peak", "intensity": 0.93}
    ]
  },
  "system_info": {
    "probable_source": "professional_camera",
    "creation_device_hints": ["Adobe Premiere Pro"]
  }
}
```

### Combined Analysis Output

The combined analysis generates the most comprehensive JSON output, containing:

```json
{
  "file_info": {
    "filename": "input_video.mp4",
    "path": "/path/to/input_video.mp4",
    "size_mb": 30.0
  },
  "technical_analysis": {
    // Full technical analysis data (as shown above)
  },
  "ai_classifications": {
    "video": [
      {"label": "Storytelling", "probability": 0.78, "count": 15},
      {"label": "Emotional", "probability": 0.65, "count": 12}
    ],
    "audio": [
      {"label": "ProfessionalRecording", "probability": 0.81, "count": 18},
      {"label": "EmotionalAudio", "probability": 0.72, "count": 14}
    ]
  },
  "combined_insights": {
    "content_type": {
      "primary_type": "STORYTELLING",
      "confidence": 0.85,
      "subtypes": ["Emotional", "Live Action"],
      "format_info": {
        "duration": 185.42,
        "resolution": "1920x1080",
        "resolution_class": "FULL_HD",
        "frame_rate": 29.97
      }
    },
    "quality_assessment": {
      "video_quality": {
        "rating": "HIGH",
        "factors": ["Professional camera work", "Good lighting and composition"]
      },
      "audio_quality": {
        "rating": "PROFESSIONAL",
        "factors": ["Clear audio with good dynamic range", "Professional post-processing"]
      },
      "technical_quality": {
        "bitrate": 2048000,
        "codec_info": {
          "video": {
            "codec": "h264",
            "profile": "High",
            "bit_depth": 8,
            "color_space": "bt709"
          },
          "audio": {
            "codec": "aac",
            "sample_rate": 48000,
            "channels": 2,
            "channel_layout": "stereo"
          }
        }
      }
    },
    "mood": {
      "primary_mood": "Emotional",
      "mood_confidence": 0.72,
      "mood_elements": [
        {"type": "Emotional", "strength": 0.72},
        {"type": "Tense", "strength": 0.28}
      ],
      "mood_consistency": "MODERATELY_CONSISTENT",
      "scene_rhythm_variation": 0.45,
      "mood_progression": [
        {"time_range": "0.0s - 37.1s", "intensity": 0.32, "action_count": 2},
        {"time_range": "37.1s - 74.2s", "intensity": 0.54, "action_count": 4},
        {"time_range": "74.2s - 111.3s", "intensity": 0.78, "action_count": 6},
        {"time_range": "111.3s - 148.4s", "intensity": 0.92, "action_count": 8},
        {"time_range": "148.4s - 185.4s", "intensity": 0.65, "action_count": 5}
      ]
    },
    "storytelling_metrics": {
      "narrative_structure": "COMPLEX",
      "pacing": "MODERATE",
      "scene_analysis": {
        "count": 42,
        "average_duration": 4.41,
        "scenes_per_minute": 13.59
      },
      "key_moments": [
        {"time": "92.70s", "description": "MAJOR", "intensity": 0.93, "type": "volume_peak"},
        {"time": "45.20s", "description": "SIGNIFICANT", "intensity": 0.85, "type": "volume_peak"},
        {"time": "132.15s", "description": "NOTABLE", "intensity": 0.68, "type": "volume_peak"}
      ]
    }
  }
}
```

### Command Line Summary Output

In addition to saving files, the tools also print a human-readable summary to the console:

```
=== Combined Media Analysis Summary ===
File: /path/to/input_video.mp4
Size: 30.00 MB
Duration: 185.42 seconds

Content Type: STORYTELLING (85.0% confidence)
Subtypes: Emotional, Live Action

Video Quality: HIGH
Audio Quality: PROFESSIONAL

Technical Highlights:
- Video: 1920x1080 @ 29.97fps (h264)
- Audio: aac 48000Hz 2ch
- Scenes: 42 (avg 4.41s)

Mood: Emotional (72.0% confidence)
Mood Consistency: MODERATELY_CONSISTENT

Narrative Structure: COMPLEX
Pacing: MODERATE

Key Moments:
- 92.70s: MAJOR (intensity: 0.93)
- 45.20s: SIGNIFICANT (intensity: 0.85)
- 132.15s: NOTABLE (intensity: 0.68)

Top Video Classifications:
- Storytelling: 0.78 (count: 15)
- Emotional: 0.65 (count: 12)
- LiveAction: 0.59 (count: 11)

Top Audio Classifications:
- ProfessionalRecording: 0.81 (count: 18)
- EmotionalAudio: 0.72 (count: 14)
- Spoken Word: 0.68 (count: 12)
```

This multi-level output format allows you to access both raw analysis data and derived insights, making it suitable for both technical and creative applications.

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

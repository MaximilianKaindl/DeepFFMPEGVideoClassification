# DeepFFmpeg Video Classification

![DeepFFmpeg Banner](https://img.shields.io/badge/DeepFFmpeg-AI%20Video%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Latest-red)

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
This project integrates modern zero-shot-classification AI models into the FFMPEG processing pipeline, offering sophisticated content analysis directly within FFmpeg.

The framework utilizes CLIP (Contrastive Language-Image Pre-training) for visual analysis and CLAP (Contrastive Language-Audio Pre-training) for audio analysis. By embedding these models as FFmpeg filters, DeepFFmpeg facilitates real-time media content analysis with a level of semantic understanding not typically available in conventional video processing tools.

Whether you need to categorize extensive media libraries, derive content insights, or develop intelligent media applications, DeepFFmpeg provides a powerful foundation for comprehending both the visual and auditory aspects of your videos with minimal overhead and maximum flexibility.

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

# Create a virtual environment with Python 3.11 specifically
python3.11 -m venv .venv

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
  --temperature 0.07 \
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
--input resources/video/example.mp4 \
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
python run_combined.py input_video.mp4  -o combined_results

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

The following example was created by the command with the default models from the conversion.
Analysis results are saved in various formats depending on the analysis type:

```bash
python run_combined.py --scene-threshold 0.1 -o combined_analysis resources/video/Popeye.mp4

# the combined script just runs those commands with added default parameters
python run_classification.py \
  --input resources/video/Popeye.mp4 \
  --scene-threshold 0.1 \
  --temperature 0.1 \
  --clip-categories resources/labels/clip_combined_analysis.txt \
  --clap-categories resources/labels/clap_combined_analysis.txt \
  --output-stats combined_analysis/Popeye_classifications.txt \
  --skip-confirmation 

python  run_analysis.py resources/video/Popeye.mp4 \
  --scene-threshold 0.1 \
  --json combined_analysis/Popeye_technical.json 
```


### Classification Output

When running visual or audio classification, results are saved in a structured CSV format:

```
stream_id,label,avg_probability,count
0,LiveAction,0.5730,20
0,HighQuality,0.5462,55
0,Informational,0.5019,93
0,Lighthearted,0.3739,2
0,Animation,0.6368,93
0,Exciting,0.3581,31
0,LowQuality,0.5495,58
0,Emotional,0.3972,76
0,RealityCapture,0.4127,12
0,Storytelling,0.4105,8
0,Tense,0.3533,2
1,BasicRecording,0.9808,113
1,InformationalAudio,0.9547,77
1,EmotionalAudio,0.9629,3
1,EerieAudio,0.9971,141
1,AnimatedAudio,0.9103,32
1,FactualAudio,0.9636,63
1,StorytellingAudio,0.9380,5
1,LightAudio,0.8344,1
```

- `stream_id`: 0 for video stream, 1 for audio stream
- `label`: The classification label from the specified categories file
- `avg_probability`: Average probability/confidence score for this label
- `count`: Number of frames/segments with this label as top classification

### Technical Analysis Output

Technical analysis generates a detailed JSON file with hierarchical information:

```json
{
  "input_file": "resources/video/Popeye.mp4",
  "metadata": {
    "format": "mov,mp4,m4a,3gp,3g2,mj2",
    "duration": 1019.319319,
    "size": 106189578,
    "bitrate": 833415,
    "streams": 2,
    "tags": {
      "major_brand": "isom",
      "minor_version": "512",
      "compatible_brands": "isomiso2avc1mp41",
      "title": "Public Domain Movies - https://archive.org/details/publicmovies212",
      "encoder": "Lavf57.21.101",
      "comment": "license:http://creativecommons.org/publicdomain/mark/1.0/"
    },
    "video_streams": [
      {
        "codec": "h264",
        "width": 720,
        "height": 480,
        "fps": 29.97,
        "bit_depth": "8",
        "pix_fmt": "yuv420p",
        "profile": "Constrained Baseline",
        "color_space": null
      }
    ],
    "audio_streams": [
      {
        "codec": "aac",
        "sample_rate": "44100",
        "channels": 2,
        "channel_layout": "stereo",
        "bit_depth": 0
      }
    ]
  },
  "scene_analysis": {
    "total_frames": 30549,
    "scene_count": 203,
    "scene_timestamps": [
      0.0,
      1.735068,
      10.076743,
      14.414414,
      // ... more timestamps
      1016.549883,
      1018.018018
    ],
    "average_scene_duration": 5.021277433497537,
    "scene_frequency": 0.199152509146155,
    "threshold_used": 0.1,
    "detection_method": "showinfo"
  },
  "audio_analysis": {
    "has_audio": true,
    "mean_volume": -26.3,
    "max_volume": -9.2,
    "action_moment_count": 39,
    "action_moments": [
      {
        "time": 7.9283220000000005,
        "type": "sound_section",
        "intensity": 1.0,
        "duration": 13.474104,
        "max_volume": null
      },
      {
        "time": 65.9318025,
        "type": "sound_section",
        "intensity": 1.0,
        "duration": 101.499977,
        "max_volume": null
      }
      // ... more moments
    ]
  },
  "system_info": {
    "classification": "Unknown system",
    "details": {
      "encoder": "Lavf57.21.101",
      "resolution_class": "SD",
      "frame_rate_class": "Broadcast standard (30fps)",
      "audio_class": "Consumer"
    }
  },
  "analysis_parameters": {
    "scene_threshold": 0.1
  }
}
```

### Combined Analysis Output

The combined analysis generates the most comprehensive JSON output, containing:

```json
{
  "file_info": {
    "filename": "Popeye.mp4",
    "path": "resources/video/Popeye.mp4",
    "size_mb": 101.27027320861816,
    "created_date": "2025-03-13 12:01:19"
  },
  "technical_analysis": {
    // Full technical analysis data (as shown above)
  },
  "ai_classifications": {
    "video": [
      {"label": "Animation", "probability": 0.7857, "count": 108},
      {"label": "LowQuality", "probability": 0.6952, "count": 106},
      {"label": "Exciting", "probability": 0.5031, "count": 101},
      {"label": "Informational", "probability": 0.4896, "count": 67},
      {"label": "Storytelling", "probability": 0.4524, "count": 43}
    ],
    "audio": [
      {"label": "EerieAudio", "probability": 0.9971, "count": 141},
      {"label": "BasicRecording", "probability": 0.9808, "count": 113},
      {"label": "InformationalAudio", "probability": 0.9547, "count": 77},
      {"label": "FactualAudio", "probability": 0.9636, "count": 63},
      {"label": "AnimatedAudio", "probability": 0.9103, "count": 32}
    ]
  },
  "combined_insights": {
    "content_type": {
      "primary_type": "Storytelling",
      "confidence": 0.16666666666666666,
      "subtypes": [
        "Animated",
        "Exciting",
        "EerieAudio"
      ],
      "format_info": {
        "duration": 1019.319319,
        "resolution": "720x480",
        "frame_rate": 29.97,
        "resolution_class": "SD"
      }
    },
    "quality_assessment": {
      "video_quality": {
        "rating": "Standard",
        "factors": [
          "Consumer-level production"
        ]
      },
      "audio_quality": {
        "rating": "Basic",
        "factors": [
          "Standard audio quality"
        ]
      },
      "technical_quality": {
        "bitrate": 833415,
        "codec_info": {
          "video": {
            "codec": "h264",
            "profile": "Constrained Baseline",
            "bit_depth": "8",
            "color_space": null
          },
          "audio": {
            "codec": "aac",
            "sample_rate": "44100",
            "channels": 2,
            "channel_layout": "stereo"
          }
        }
      }
    },
    "mood": {
      "primary_mood": "Eerie",
      "mood_confidence": 0.7080142619012847,
      "mood_elements": [
        {"type": "Exciting", "strength": 0.255893861641428},
        {"type": "Emotional", "strength": 0.025633652446731898},
        {"type": "Tense", "strength": 0.004131519708315917},
        {"type": "Lighthearted", "strength": 0.0021246808446349166},
        {"type": "Eerie", "strength": 0.7080142619012847},
        {"type": "Light", "strength": 0.004202023457604585}
      ],
      "mood_progression": [
        {"time_range": "0.0s - 203.9s", "intensity": 0.8818211714285715, "action_count": 7},
        {"time_range": "203.9s - 407.7s", "intensity": 0.6815447749999997, "action_count": 8},
        {"time_range": "407.7s - 611.6s", "intensity": 0, "action_count": 0},
        {"time_range": "611.6s - 815.5s", "intensity": 0, "action_count": 0},
        {"time_range": "815.5s - 1019.3s", "intensity": 0, "action_count": 0}
      ],
      "mood_consistency": "Moderately consistent",
      "scene_rhythm_variation": 0.4137186029823504
    },
    "storytelling_metrics": {
      "scene_analysis": {
        "count": 203,
        "average_duration": 5.021277433497537,
        "scenes_per_minute": 11.949150548769301
      },
      "key_moments": [
        {"time": "7.93s", "description": "Major action/emotional moment", "intensity": 1.0, "type": "sound_section"},
        {"time": "65.93s", "description": "Major action/emotional moment", "intensity": 1.0, "type": "sound_section"},
        {"time": "127.04s", "description": "Major action/emotional moment", "intensity": 1.0, "type": "sound_section"}
      ],
      "narrative_structure": "Rising Action",
      "pacing": "Fast"
    }
  }
}
```

### Command Line Summary Output

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

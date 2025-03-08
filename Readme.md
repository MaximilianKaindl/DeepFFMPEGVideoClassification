# DeepFFmpeg Video Classification

![DeepFFmpeg Banner](https://img.shields.io/badge/DeepFFmpeg-AI%20Video%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Latest-red)

DeepFFmpeg is a powerful, modular framework that integrates state-of-the-art deep learning models with FFmpeg to enable advanced video and audio analysis capabilities. This project allows you to perform object detection, visual content classification, and audio classification directly within the FFmpeg pipeline.

## 🔍 Features

- **Multi-modal Analysis**: Combine YOLO object detection, CLIP visual understanding, and CLAP audio classification for comprehensive media analysis
- **Scene Detection**: Analyze only significant scene changes to improve performance and accuracy
- **Flexible Pipelines**: Create custom analysis pipelines with different models and configurations
- **Output Options**: Generate annotated videos and detailed analysis logs
- **GPU Acceleration**: Leverage CUDA for faster processing
- **Model Conversion**: Automated tools to convert and test CLIP and CLAP models

## 📋 Requirements

- Python 3.10+
- FFmpeg with custom DNN modules (see [FFmpeg fork](#ffmpeg-fork))
- LibTorch C++ libraries
- tokenizers-cpp
- OpenVINO Toolkit
- Additional FFmpeg development libraries
- GPU support (optional but recommended)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MaximilianKaindl/DeepFFMPEGVideoClassification.git
cd DeepFFMPEGVideoClassification
```

### 2. Install FFmpeg dependencies

```bash
# Install FFmpeg development libraries
sudo apt install -y libass-dev
sudo apt install -y libfdk-aac-dev
sudo apt install -y libmp3lame-dev
sudo apt install -y libopus-dev
sudo apt-get install -y libvpx-dev
sudo apt-get install -y libx264-dev
sudo apt-get install -y libx265-dev
sudo apt-get install -y libsdl2-dev
sudo apt install nvida-cuda-toolkit
```

### 3. Install LibTorch and tokenizers-cpp

```bash
# Download and extract LibTorch (C++ libraries)
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu126.zip -d /path/to/install

# Clone and build tokenizers-cpp
git clone https://github.com/mlc-ai/tokenizers-cpp.git
cd tokenizers-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..
```
Download Openvino Toolkit
https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/linux

The existing FFMPEG binary was configured like this. If shared libraries are missing, the program wont work. 
If desired, the program can be confiured differently using my fork listed below.

#### Then set Environment Variables with `set_vars.sh`

- `LIBTORCH_ROOT`: Path to LibTorch installation
- `TOKENIZER_ROOT`: Path to tokenizer installation
- `OPENVINO_ROOT`: Path to OpenVINO installation

These variables must be set correctly for the FFmpeg modules to work properly. The configuration script uses these variables to locate the necessary headers and libraries during compilation.


### 4. Set up the Python environment

Using Conda (recommended):
```bash
conda env create -f environment.yml
conda activate deepffmpegvideoclassification
```

Or using pip:
```bash
python -m pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
# Edit the paths in set_vars.sh to point to your installations
# The variables in this file are critical for both building FFmpeg and running the scripts
# LIBTORCH_ROOT, TOKENIZER_ROOT, and OPENVINO_ROOT must be set correctly
vim set_vars.sh  # Edit paths as needed

# Then source the file
source ./set_vars.sh
```

### 6. Download and convert YOLO models (if needed)

```bash
# Install required packages
pip install openvino-dev tensorflow

# Download YOLO model
omz_downloader --name yolo-v4-tiny-tf

# Convert the model to OpenVINO format
omz_converter --name yolo-v4-tiny-tf

# Download COCO class labels
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/refs/heads/master/data/dataset_classes/coco_80cl.txt -O resources/labels/coco_80cl.txt
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
```

## 🔧 Usage

### Running a basic visual analysis with CLIP

```bash
python run_ffmpeg.py \
  --input resources/video/example.mp4 \
  --scene-threshold 0.4 \
  --clip-model models/clip/clip_vit_b_32.pt \
  --categories resources/labels/categories_clip.txt \
  --tokenizer models/clip/tokenizer_clip/tokenizer.json \
```

### Running audio analysis with CLAP

```bash
python run_ffmpeg.py \
  --input resources/audio/sample.mp3 \
  --device cpu \
  --clap-model models/clap/msclap2023.pt \
  --audio-labels resources/labels/categories_clap.txt \
  --audio-tokenizer models/clap/tokenizer_clap/tokenizer.json \
```

### Complete pipeline with detection, CLIP and CLAP

```bash
python run_ffmpeg.py \
  --input resources/video/sample.mp4 \
  --scene-threshold 0.4 \
  --confidence 0.1 \
  --detect-model models/detect/public/yolo-v4-tiny-tf/FP16/yolo-v4-tiny-tf.xml \
  --labels resources/labels/coco_80cl.txt \
  --anchors "81&82&135&169&344&319" \
  --nb-classes 80 \
  --clip-model models/clip/clip_vit_b_32.pt \
  --tokenizer models/clip/tokenizer_clip/tokenizer.json \
  --labels resources/labels/labels_clip_person.txt \
  --clap-model models/clap/msclap2023.pt \
  --audio-tokenizer models/clap/tokenizer_clap/tokenizer.json \
  --output-video output/annotated_video.mp4
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
└── set_vars.sh                  # Environment setup script
```

## 🔬 Configuration

### Custom Labels and Categories

You can create custom category files for both CLIP and CLAP models:

```
# Example CLIP categories file
person
car
building
tree
sky
water
```

## ⚠️ Known Issues and Troubleshooting

- **Missing Libraries**: If you encounter errors about missing libraries during FFmpeg compilation, make sure all development packages are installed correctly.
- **CUDA Compatibility**: Ensure your CUDA version matches the LibTorch build you downloaded.
- **Path Configuration**: Double-check the paths in `set_vars.sh` to ensure they point to the correct installations. These paths are used by both the build process and runtime scripts.
- **Missing Dependencies**: You might need to install additional dependencies such as `libfreetype`, `libfontconfig`, and `libxcb` development packages.
- **CUDA Configuration**: Make sure CUDA is properly installed if you're enabling CUDA features with `--enable-cuda-nvcc` and `--enable-libnpp`.
- **Model Conversion Failures**: If model conversion fails, try running with the `--verbose` flag for more detailed error messages.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔄 FFmpeg Fork

This project uses a custom fork of FFmpeg that includes additional DNN modules for CLIP and CLAP model integration. The fork adds specialized filters that enable deep learning-based video and audio analysis directly within the FFmpeg pipeline.

```bash
# Clone the FFmpeg fork
git clone https://github.com/MaximilianKaindl/FFmpeg.git

# Configure and build with required modules
cd ffmpeg-deepclassification
./configure \
    --enable-gpl \
    --enable-debug \
    --enable-openssl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libfdk-aac \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libass \
    --enable-libfreetype \
    --enable-libfontconfig \
    --enable-libxcb \
    --enable-sdl2 \
    --enable-libopenvino \
    --enable-libtorch \
    --enable-libtokenizers \
    --enable-cuda-nvcc \
    --enable-libnpp \
    --enable-nonfree \
    --extra-cflags="-I$LIBTORCH_HEADER \
                    -I$LIBTORCH_HEADER_CSRC \
                    -I$TOKENIZER_HEADER \
                    -I$OPENVINO_HEADER" \
    --extra-ldflags="-L$LIBTORCH_LIB \
                     -L$TOKENIZER_LIB \
                     -L$OPENVINO_LIB"

make -j$(nproc)
sudo make install
```

> **Note:** Make sure to set the correct paths to LibTorch, tokenizers-cpp, and OpenVINO in your `set_vars.sh` file before building FFmpeg.

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

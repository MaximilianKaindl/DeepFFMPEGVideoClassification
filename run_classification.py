#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import json

class FFmpegCommandBuilder:
    def __init__(self):
        # Default directories
        self.models_dir = "models"
        self.resources_dir = "resources"
        self.ffmpeg_path = "./FFmpeg/ffmpeg"  # Assuming ffmpeg binary is in current directory
        
        # Load model configuration if it exists
        self.model_config = self.load_model_config()

        # Environment variables
        self.env_vars = {
            "LIBTORCH_ROOT": os.environ.get("LIBTORCH_ROOT", ""),
            "TOKENIZER_ROOT": os.environ.get("TOKENIZER_ROOT", ""),
            "OPENVINO_ROOT": os.environ.get("OPENVINO_ROOT", "")
        }

        # Check environment variables
        self.check_env_vars()

    def load_model_config(self):
        """Load model configuration from models_config.json if it exists"""
        config_path = os.path.join(self.models_dir, "models_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model configuration: {e}")
        return None

    def check_env_vars(self):
        """Check if required environment variables are set"""
        missing_vars = [var for var, value in self.env_vars.items() if not value]
        
        if missing_vars:
            print("Warning: Some Environment variables are missing. \n Ignore this message if you sourced the setup file and use the same terminal. \n Setup environment variables with setup.sh in /FFmpeg. FFMPEG is configured to run with these variables. FFMPEG Build may fail if these variables are not set.")

    def check_dependencies(self):
        """Check if necessary tools and libraries are installed"""
        # Check for FFmpeg
        if not shutil.which(self.ffmpeg_path) and not shutil.which("ffmpeg"):
            print("Error: FFmpeg not found. Please install FFmpeg or set the correct path.")
            return False
            
        # Check for Python packages
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
        except ImportError:
            print("Warning: PyTorch not found. CLIP and CLAP models may not work.")
            
        return True

    def build_detection_filter(self, args):
        """Build the FFmpeg detection filter string"""
        if not args.detect_model:
            return ""
            
        # Default YOLO parameters
        model_type = "yolov4"
        confidence = args.confidence if args.confidence else 0.1
        labels_path = args.detect_labels if args.detect_labels else "resources/labels/coco_80cl.txt"
        anchors = args.anchors if args.anchors else "81&82&135&169&344&319"
        nb_classes = args.nb_classes if args.nb_classes else 80
        
        detection_filter = f"dnn_detect=dnn_backend=openvino:model={args.detect_model}:confidence={confidence}"
        detection_filter += f":labels={labels_path}:async=0:nb_classes={nb_classes}"
        detection_filter += f":model_type={model_type}:anchors={anchors}"
        
        return detection_filter

    def build_clip_filter(self, args):
        """Build the FFmpeg CLIP classification filter string"""
        if not args.clip_model:
            return ""
            
        # Check if we have specified CLIP parameters
        conf = args.confidence if args.confidence else 0.5
        tokenizer = args.tokenizer
        
        # If tokenizer not specified but we have model_config, use tokenizer from config
        if not tokenizer and self.model_config and "clip_model" in self.model_config:
            tokenizer = self.model_config["clip_model"]["tokenizer_dir"]
            if tokenizer:
                print(f"Using CLIP tokenizer from config: {tokenizer}")
        
        # Determine if we're using categories or labels
        clip_filter = f"dnn_classify=dnn_backend=torch:model={args.clip_model}"
        clip_filter += f":device={args.device}:confidence={conf}"
        
        # Add temperature and logit_scale if specified
        # First try model-specific parameters, then fall back to common parameters
        if args.clip_temperature is not None:
            clip_filter += f":temperature={args.clip_temperature}"
        elif args.temperature is not None:
            clip_filter += f":temperature={args.temperature}"
        
        if args.clip_logit_scale is not None:
            clip_filter += f":logit_scale={args.clip_logit_scale}"
        elif args.logit_scale is not None:
            clip_filter += f":logit_scale={args.logit_scale}"
        
        if tokenizer:
            clip_filter += f":tokenizer={tokenizer}"
        
        if args.clip_categories:
            clip_filter += f":categories={args.clip_categories}"
        elif args.clip_labels:
            clip_filter += f":labels={args.clip_labels}"
        else:
            print("Warning: No categories or labels specified for CLIP. Using default.")
            clip_filter += f":labels=resources/labels/labels_clip_animal.txt"

        # Add target if specified
        if args.target:
            clip_filter += f":target={args.target}"
            
        return clip_filter

    def build_clap_filter(self, args):
        """Build the FFmpeg CLAP audio classification filter string"""
        if not args.clap_model:
            return ""
            
        conf = args.confidence if args.confidence else 0.5
        tokenizer = args.audio_tokenizer
        
        # Determine which device to use for CLAP
        clap_device = args.device  # Default to user-specified device
        
        # If tokenizer not specified but we have model_config, use tokenizer from config
        if self.model_config and "clap_model" in self.model_config:
            # Get tokenizer from config if not specified
            if not tokenizer:
                tokenizer = self.model_config["clap_model"].get("tokenizer_path")
                if tokenizer:
                    print(f"Using CLAP tokenizer from config: {tokenizer}")
            
            # Check if device_traced is specified in config
            device_traced = self.model_config["clap_model"].get("device_traced")
            if device_traced == "cuda":
                if args.device != "cuda":
                    print("Warning: CLAP model was traced with CUDA but requested device is CPU.")
                    print("Using CUDA for CLAP as it's required for models traced with CUDA.")
                clap_device = "cuda"
        
        # Build the filter with correct device
        clap_filter = f"dnn_classify=dnn_backend=torch:is_audio=1:device={clap_device}:model={args.clap_model}"
        
        # Add temperature and logit_scale if specified
        # First try model-specific parameters, then fall back to common parameters
        if args.clap_temperature is not None:
            clap_filter += f":temperature={args.clap_temperature}"
        elif args.temperature is not None:
            clap_filter += f":temperature={args.temperature}"
        
        if args.clap_logit_scale is not None:
            clap_filter += f":logit_scale={args.clap_logit_scale}"
        elif args.logit_scale is not None:
            clap_filter += f":logit_scale={args.logit_scale}"
        
        if tokenizer:
            clap_filter += f":tokenizer={tokenizer}"
            
        clap_filter += f":confidence={conf}"

        # Determine if we're using categories or labels for audio
        if args.clap_categories:
            clap_filter += f":categories={args.clap_categories}"
        elif args.clap_labels:
            clap_filter += f":labels={args.clap_labels}"
        else:
            # Default case
            print("Warning: No categories or labels specified for CLAP. Using default.")
            clap_filter += f":labels=resources/labels/labels_clap_music.txt"
            
        return clap_filter

    def build_average_filter(self, args):
        """Build the FFmpeg average classification filter string"""
        avg_filter = "avgclass="
        
        # Add video stream count
        use_video = args.detect_model or args.clip_model
        avg_filter += f"v={1 if use_video else 0}"
        
        # Add audio stream count
        use_audio = args.clap_model
        avg_filter += f":a={1 if use_audio else 0}"
        
        # Add output file if specified
        if args.output_stats:
            avg_filter += f":output_file={args.output_stats}"
            
        return avg_filter

    def apply_model_defaults(self, args):
        """Apply defaults from model configuration if arguments are not provided"""
        if not self.model_config:
            return args

        clip_input = args.clip_categories or args.clip_labels
        clap_input = args.clap_categories or args.clap_labels
        
        # Apply CLIP model defaults
        if clip_input and not args.clip_model and self.model_config.get("clip_model", {}).get("path"):
            clip_path = self.model_config["clip_model"]["path"]
            if os.path.exists(clip_path):
                args.clip_model = clip_path
                print(f"Using default CLIP model from config: {clip_path}")
                
                # Also set tokenizer if not provided
                if not args.tokenizer and self.model_config["clip_model"].get("tokenizer_path"):
                    args.tokenizer = self.model_config["clip_model"]["tokenizer_path"]
                    print(f"Using default CLIP tokenizer from config: {args.tokenizer}")
        
        # Apply CLAP model defaults
        if clap_input and not args.clap_model and self.model_config.get("clap_model", {}).get("path"):
            clap_path = self.model_config["clap_model"]["path"]
            if os.path.exists(clap_path):
                args.clap_model = clap_path
                print(f"Using default CLAP model from config: {clap_path}")
                
                # Also set tokenizer if not provided
                if not args.audio_tokenizer and self.model_config["clap_model"].get("tokenizer_path"):
                    args.audio_tokenizer = self.model_config["clap_model"]["tokenizer_path"]
                    print(f"Using default CLAP tokenizer from config: {args.audio_tokenizer}")
        
        return args

    def build_command(self, args):
        """Build the complete FFmpeg command based on arguments"""
        # Apply defaults from model configuration
        args = self.apply_model_defaults(args)
        
        # Base command
        cmd = [self.ffmpeg_path]
        
        # Add verbosity
        if args.verbose:
            cmd.extend(["-v", "debug"])
        else:
            cmd.extend(["-v", "info"])
            
        # Input file
        cmd.extend(["-i", args.input])
        
        # Build filter complex or filter graph
        if args.clap_model:
            # If audio analysis is needed, we have to use filter_complex
            filter_complex = []
            
            # Add audio stream processing
            audio_filter = f"[0:a] {self.build_clap_filter(args)} [audio]"
            filter_complex.append(audio_filter)
            
            # Add video stream processing if needed
            if args.detect_model or args.clip_model:
                # Only add scene filter if threshold is positive
                scene_filter = f"select='gt(scene,{args.scene_threshold})'," if args.scene_threshold > 0 else ""
                
                if args.detect_model and args.clip_model:
                    # Both detection and CLIP
                    video_filter = f"[0:v] {scene_filter}{self.build_detection_filter(args)},{self.build_clip_filter(args)} [video]"
                elif args.detect_model:
                    # Detection only
                    video_filter = f"[0:v] {scene_filter}{self.build_detection_filter(args)} [video]"
                else:
                    # CLIP only
                    video_filter = f"[0:v] {scene_filter}{self.build_clip_filter(args)} [video]"
                    
                filter_complex.append(video_filter)
                
                # Combine video and audio
                filter_complex.append(f"[video][audio] {self.build_average_filter(args)}")
            else:
                # Only audio analysis
                filter_complex.append(f"[audio] {self.build_average_filter(args)}")
                
            cmd.extend(["-filter_complex", ";".join(filter_complex)])
        else:
            # Video-only processing uses -vf
            filters = []
            
            # Only add scene detection if threshold is positive
            if args.scene_threshold > 0:
                filters.append(f"select='gt(scene,{args.scene_threshold})'")
                
            # Add detection if needed
            if args.detect_model:
                filters.append(self.build_detection_filter(args))
                
            # Add CLIP if needed
            if args.clip_model:
                filters.append(self.build_clip_filter(args))
                
            # Add average classification
            filters.append(self.build_average_filter(args))
            
            cmd.extend(["-vf", ",".join(filters)])

        cmd.extend(["-f", "null", "-"])
        return cmd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build FFmpeg commands for AI-based video/audio analysis')
    
    # Input/output options
    parser.add_argument('--input', required=True, help='Input video/audio file')
    parser.add_argument('--output-stats', help='Output statistics file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Scene detection
    parser.add_argument('--scene-threshold', type=float, default=-1.0, 
                        help='Scene change threshold')
    
    # Confidence threshold
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold for detections and classifications (default: 0.3)')
    
    # Device selection
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for CLIP and CLAP models (default: cuda)')
    
    # Common model parameters group (kept for backward compatibility)
    model_params_group = parser.add_argument_group('Common Model Parameters')
    model_params_group.add_argument('--temperature', type=float, 
                                   help='Softmax temperature for both CLIP and CLAP (higher = smoother probabilities)')
    model_params_group.add_argument('--logit-scale', type=float, 
                                   help='Logit scale for both CLIP and CLAP similarity calculation')
    
    # YOLO detection options
    detection_group = parser.add_argument_group('YOLO Detection Options')
    detection_group.add_argument('--detect-model', 
                                help='Path to YOLO detection model (.xml)')
    detection_group.add_argument('--detect-labels', 
                                help='Path to labels file for detection')
    detection_group.add_argument('--anchors', 
                                help='Anchor values for YOLO (comma-separated)')
    detection_group.add_argument('--nb-classes', type=int, 
                                help='Number of classes for YOLO model')
    
    # CLIP options
    clip_group = parser.add_argument_group('CLIP Classification Options')
    clip_group.add_argument('--clip-model', 
                           help='Path to CLIP model (.pt)')
    
    # Create mutually exclusive group for CLIP labels/categories
    clip_labels_group = clip_group.add_mutually_exclusive_group()
    clip_labels_group.add_argument('--clip-categories', 
                                  help='Path to categories file for CLIP')
    clip_labels_group.add_argument('--clip-labels', 
                                  help='Path to labels file for CLIP')
    
    clip_group.add_argument('--tokenizer', 
                           help='Path to tokenizer file for CLIP')
    clip_group.add_argument('--target', 
                           help='Target object to classify')
    clip_group.add_argument('--clip-temperature', type=float,
                           help='Softmax temperature for CLIP (higher = smoother probabilities)')
    clip_group.add_argument('--clip-logit-scale', type=float,
                           help='Logit scale for CLIP similarity calculation')
    
    # CLAP options
    clap_group = parser.add_argument_group('CLAP Audio Classification Options')
    clap_group.add_argument('--clap-model', 
                           help='Path to CLAP model (.pt)')
    
    # Create mutually exclusive group for CLAP labels/categories
    clap_labels_group = clap_group.add_mutually_exclusive_group()
    clap_labels_group.add_argument('--clap-categories', 
                                  help='Path to categories file for CLAP')
    clap_labels_group.add_argument('--clap-labels', 
                                  help='Path to labels file for CLAP')
    
    clap_group.add_argument('--audio-tokenizer', 
                           help='Path to tokenizer file for CLAP')
    clap_group.add_argument('--clap-temperature', type=float,
                           help='Softmax temperature for CLAP (higher = smoother probabilities)')
    clap_group.add_argument('--clap-logit-scale', type=float,
                           help='Logit scale for CLAP similarity calculation')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    builder = FFmpegCommandBuilder()
    
    if not builder.check_dependencies():
        sys.exit(1)
        
    cmd = builder.build_command(args)
    
    # Print the command
    print("\nGenerated FFmpeg command:")
    print(" ".join(cmd))
    
    # Ask user if they want to execute the command
    response = input("\nDo you want to execute this command? (y/n): ")
    if response.lower() in ['y', 'yes']:
        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error executing command: {e}")
            sys.exit(1)
    else:
        print("Command not executed. You can copy and run it manually.")

if __name__ == "__main__":
    main()
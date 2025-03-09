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

        # Environment variables
        self.env_vars = {
            "LIBTORCH_ROOT": os.environ.get("LIBTORCH_ROOT", ""),
            "TOKENIZER_ROOT": os.environ.get("TOKENIZER_ROOT", ""),
            "OPENVINO_ROOT": os.environ.get("OPENVINO_ROOT", "")
        }

        # Check environment variables
        self.check_env_vars()

    def check_env_vars(self):
        """Check if required environment variables are set"""
        missing_vars = [var for var, value in self.env_vars.items() if not value]
        
        if missing_vars:
            print("Warning: Setup environment variables from set_vars.sh. FFMPEG is configured to run with these variables. FFMPEG Build may fail if these variables are not set.")

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
        labels_path = args.labels if args.labels else "resources/labels/coco_80cl.txt"
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
        conf = args.confidence if args.confidence else 0.05
        
        # Determine if we're using categories or labels
        clip_filter = f"dnn_classify=dnn_backend=torch:model={args.clip_model}"
        clip_filter += f":device={args.device}:confidence={conf}:tokenizer={args.tokenizer}"
        
        if args.categories:
            clip_filter += f":categories={args.categories}"
        elif args.labels:
            clip_filter += f":labels={args.labels}"
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
            
        conf = args.confidence if args.confidence else 0.05
        clap_filter = f"dnn_classify=dnn_backend=torch:is_audio=1:device={args.device}:model={args.clap_model}:tokenizer={args.audio_tokenizer}:confidence={conf}"

        # Determine if we're using categories or labels for audio
        if args.audio_categories:
            clap_filter += f":categories={args.audio_categories}"
        elif args.audio_labels:
            clap_filter += f":labels={args.audio_labels}"
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

    def build_command(self, args):
        """Build the complete FFmpeg command based on arguments"""
        # Check for required models
        
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
                scene_filter = f"select='gt(scene,{args.scene_threshold})'"
                
                if args.detect_model and args.clip_model:
                    # Both detection and CLIP
                    video_filter = f"[0:v] {scene_filter},{self.build_detection_filter(args)},{self.build_clip_filter(args)} [video]"
                elif args.detect_model:
                    # Detection only
                    video_filter = f"[0:v] {scene_filter},{self.build_detection_filter(args)} [video]"
                else:
                    # CLIP only
                    video_filter = f"[0:v] {scene_filter},{self.build_clip_filter(args)} [video]"
                    
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
            
            # Scene detection
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
    parser.add_argument('--scene-threshold', type=float, default=0.4, 
                        help='Scene change threshold (default: 0.4)')
    
    # Confidence threshold
    parser.add_argument('--confidence', type=float, default=0.05,
                        help='Confidence threshold for detections and classifications (default: 0.05)')
    
    # Device selection
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for CLIP and CLAP models (default: cuda)')
    
    # YOLO detection options
    detection_group = parser.add_argument_group('YOLO Detection Options')
    detection_group.add_argument('--detect-model', 
                                help='Path to YOLO detection model (.xml)')
    detection_group.add_argument('--labels', 
                                help='Path to labels file for detection')
    detection_group.add_argument('--anchors', 
                                help='Anchor values for YOLO (comma-separated)')
    detection_group.add_argument('--nb-classes', type=int, 
                                help='Number of classes for YOLO model')
    
    # CLIP options
    clip_group = parser.add_argument_group('CLIP Classification Options')
    clip_group.add_argument('--clip-model', 
                           help='Path to CLIP model (.pt)')
    clip_group.add_argument('--categories', 
                           help='Path to categories file for CLIP')
    clip_group.add_argument('--tokenizer', 
                           help='Path to tokenizer file for CLIP')
    clip_group.add_argument('--target', 
                           help='Target object to classify')
    
    # CLAP options
    clap_group = parser.add_argument_group('CLAP Audio Classification Options')
    clap_group.add_argument('--clap-model', 
                           help='Path to CLAP model (.pt)')
    clap_group.add_argument('--audio-categories', 
                           help='Path to categories file for CLAP')
    clap_group.add_argument('--audio-labels', 
                           help='Path to labels file for CLAP')
    clap_group.add_argument('--audio-tokenizer', 
                           help='Path to tokenizer file for CLAP')
    
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
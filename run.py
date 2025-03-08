#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import json
from pathlib import Path

def setup_directories(args):
    """Create required directories if they don't exist."""
    directories = [
        args.models_dir,
        args.clip_model_dir,
        args.clap_model_dir,
        args.clip_tokenizer_dir,
        args.clap_tokenizer_dir,
        args.resources_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")

def convert_clip_model(args):
    """Convert CLIP model if it doesn't exist."""
    model_path = os.path.join(args.clip_model_dir, f"clip_{args.clip_model_name.replace('-', '_').lower()}.pt")
    tokenizer_info_path = os.path.join(args.clip_tokenizer_dir, "tokenizer_info.json")
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_info_path):
        print(f"CLIP model already exists at {model_path}")
        print(f"CLIP tokenizer already exists at {args.clip_tokenizer_dir}")
        return True
    
    # Convert model
    print(f"\n=== Converting CLIP model ({args.clip_model_name}) ===")
    cmd = [
        sys.executable, "converters/clip_to_pt.py",
        "--model_name", args.clip_model_name,
        "--dataset_name", args.clip_dataset,
        "--output_path", model_path,
        "--tokenizer_dir", args.clip_tokenizer_dir
    ]
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        
        if os.path.exists(model_path):
            print(f"Successfully converted CLIP model: {model_path}")
            print(f"Model size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print("CLIP model conversion failed - output file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"CLIP model conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def convert_clap_model(args):
    """Convert CLAP model if it doesn't exist."""
    model_path = os.path.join(args.clap_model_dir, f"msclap{args.clap_version}.pt")
    tokenizer_config_path = os.path.join(args.clap_tokenizer_dir, "config.json")
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_config_path):
        print(f"CLAP model already exists at {model_path}")
        print(f"CLAP tokenizer already exists at {args.clap_tokenizer_dir}")
        return True
    
    # Convert model
    print(f"\n=== Converting CLAP model (version {args.clap_version}) ===")
    cmd = [
        sys.executable, "converters/clap_to_pt.py",
        "--version", args.clap_version,
        "--output_dir", args.clap_model_dir,
        "--tokenizer_dir", args.clap_tokenizer_dir,
        "--audio_path", args.audio_path
    ]
    
    if args.use_cuda:
        cmd.append("--use_cuda")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        
        if os.path.exists(model_path):
            print(f"Successfully converted CLAP model: {model_path}")
            print(f"Model size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print("CLAP model conversion failed - output file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"CLAP model conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def save_config(args):
    """Save configuration data for future reference."""
    config = {
        "clip_model": {
            "name": args.clip_model_name,
            "dataset": args.clip_dataset,
            "path": os.path.join(args.clip_model_dir, f"clip_{args.clip_model_name.replace('-', '_').lower()}.pt"),
            "tokenizer_dir": args.clip_tokenizer_dir
        },
        "clap_model": {
            "version": args.clap_version,
            "path": os.path.join(args.clap_model_dir, f"traced_clap_{args.clap_version}.pt"),
            "tokenizer_dir": args.clap_tokenizer_dir
        },
        "resources": {
            "audio_path": args.audio_path
        }
    }
    
    config_path = os.path.join(args.models_dir, "models_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert CLIP and CLAP models if they don't exist")
    
    # Directory structure arguments
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Base directory for all models")
    parser.add_argument("--clip_model_dir", type=str, default="models/clip",
                        help="Directory for CLIP models")
    parser.add_argument("--clap_model_dir", type=str, default="models/clap",
                        help="Directory for CLAP models")
    parser.add_argument("--clip_tokenizer_dir", type=str, default="models/clip/tokenizer_clip",
                        help="Directory for CLIP tokenizer")
    parser.add_argument("--clap_tokenizer_dir", type=str, default="models/clap/tokenizer_clap",
                        help="Directory for CLAP tokenizer")
    parser.add_argument("--resources_dir", type=str, default="resources",
                        help="Directory for resource files")
    
    # CLIP model arguments
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32",
                        help="CLIP model name")
    parser.add_argument("--clip_dataset", type=str, default="laion2b_s34b_b79k",
                        help="CLIP dataset name")
    parser.add_argument("--image_path", type=str, default="resources/images/cat.jpg",
                        help="Path to iamge file for CLIP testing")
    # CLAP model arguments
    parser.add_argument("--clap_version", type=str, default="2023",
                        choices=["2022", "2023", "clapcap"],
                        help="CLAP model version")
    parser.add_argument("--audio_path", type=str, default="resources/audio/blues.mp3",
                        help="Path to audio file for CLAP tracing")
    
    # General arguments
    parser.add_argument("--skip_clip", action="store_true",
                        help="Skip CLIP model conversion")
    parser.add_argument("--skip_clap", action="store_true",
                        help="Skip CLAP model conversion")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA for model conversion if available")
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories(args)
    
    # Convert models
    clip_success = True
    clap_success = True
    
    if not args.skip_clip:
        clip_success = convert_clip_model(args)
    
    if not args.skip_clap:
        clap_success = convert_clap_model(args)
    
    # Save configuration
    if clip_success and clap_success:
        save_config(args)
        print("\nAll models converted successfully!")
    else:
        print("\nSome model conversions failed. Check the logs above for details.")
    
if __name__ == "__main__":
    main()
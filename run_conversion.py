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
    tokenizer_path = args.clip_tokenizer_dir
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print(f"CLIP model already exists at {model_path}")
        print(f"CLIP tokenizer already exists at {args.clip_tokenizer_dir}")
        return model_path
    
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
            return model_path
        else:
            print("CLIP model conversion failed - output file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"CLIP model conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def convert_clap_model(args):
    """Convert CLAP model if it doesn't exist."""
    model_path = os.path.join(args.clap_model_dir, f"msclap{args.clap_version}.pt")
    tokenizer_path = args.clap_tokenizer_dir
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print(f"CLAP model already exists at {model_path}")
        print(f"CLAP tokenizer already exists at {args.clap_tokenizer_dir}")
        return model_path
    
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
            return model_path
        else:
            print("CLAP model conversion failed - output file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"CLAP model conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def test_clip_model(args, model_path):
    """Test the converted CLIP model."""
    if not model_path:
        return False
        
    print(f"\n=== Testing CLIP model ({args.clip_model_name}) ===")
    
    # Define output file for results
    results_dir = os.path.join(args.models_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"clip_{args.clip_model_name.replace('-', '_').lower()}_results.json")
    
    # Build command
    cmd = [
        sys.executable, "converters/test_scripted_models.py",
        "--model_path", model_path,
        "--model_type", "clip",
        "--image_path", args.image_path,
        "--clip_model_name", args.clip_model_name,
        "--output_file", output_file
    ]
    
    if args.use_cuda:
        cmd.append("--use_cuda")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        
        if os.path.exists(output_file):
            print(f"CLIP model test results saved to {output_file}")
            return True
        else:
            print("CLIP model testing completed, but no results file was generated")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"CLIP model testing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_clap_model(args, model_path):
    """Test the converted CLAP model."""
    if not model_path:
        return False
        
    print(f"\n=== Testing CLAP model (version {args.clap_version}) ===")
    
    # Define output file for results
    results_dir = os.path.join(args.models_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"clap_{args.clap_version}_results.json")
    
    # Build command
    cmd = [
        sys.executable, "converters/test_scripted_models.py",
        "--model_path", model_path,
        "--model_type", "clap",
        "--audio_path", args.audio_path,
        "--output_dir", args.clap_tokenizer_dir,
        "--output_file", output_file,
        "--sample_rate", str(44100),  # Default sample rate
        "--duration", str(7.0)       # Default duration
    ]
    
    if args.use_cuda:
        cmd.append("--use_cuda")
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        
        if os.path.exists(output_file):
            print(f"CLAP model test results saved to {output_file}")
            return True
        else:
            print("CLAP model testing completed, but no results file was generated")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"CLAP model testing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def save_config(args, test_results):
    """Save configuration data for future reference."""
    config = {
        "clip_model": {
            "name": args.clip_model_name,
            "dataset": args.clip_dataset,
            "path": os.path.join(args.clip_model_dir, f"clip_{args.clip_model_name.replace('-', '_').lower()}.pt"),
            "tokenizer_dir": args.clip_tokenizer_dir,
            "tested": test_results.get("clip", False)
        },
        "clap_model": {
            "version": args.clap_version,
            "path": os.path.join(args.clap_model_dir, f"msclap{args.clap_version}.pt"),
            "tokenizer_dir": args.clap_tokenizer_dir,
            "tested": test_results.get("clap", False)
        },
        "resources": {
            "audio_path": args.audio_path,
            "image_path": args.image_path
        }
    }
    
    config_path = os.path.join(args.models_dir, "models_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert and test CLIP and CLAP models")
    
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
    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14",
                        help="CLIP model name")
    parser.add_argument("--clip_dataset", type=str, default="datacomp_xl_s13b_b90k",
                        help="CLIP dataset name")
    parser.add_argument("--image_path", type=str, default="resources/images/cat.jpg",
                        help="Path to image file for CLIP testing")
    
    # CLAP model arguments
    parser.add_argument("--clap_version", type=str, default="2023",
                        choices=["2022", "2023", "clapcap"],
                        help="CLAP model version")
    parser.add_argument("--audio_path", type=str, default="resources/audio/blues.mp3",
                        help="Path to audio file for CLAP tracing and testing")
    
    # General arguments
    parser.add_argument("--skip_clip", action="store_true",
                        help="Skip CLIP model conversion and testing")
    parser.add_argument("--skip_clap", action="store_true",
                        help="Skip CLAP model conversion and testing")
    parser.add_argument("--skip_tests", action="store_true",
                        help="Skip testing of converted models")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA for model conversion and testing if available")
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories(args)
    
    # Track test results
    test_results = {}
    
    # Convert and test CLIP model
    clip_model_path = None
    if not args.skip_clip:
        clip_model_path = convert_clip_model(args)
        if not args.skip_tests and clip_model_path:
            test_results["clip"] = test_clip_model(args, clip_model_path)
    
    # Convert and test CLAP model
    clap_model_path = None
    if not args.skip_clap:
        clap_model_path = convert_clap_model(args)
        if not args.skip_tests and clap_model_path:
            test_results["clap"] = test_clap_model(args, clap_model_path)
    
    # Save configuration
    save_config(args, test_results)
    
    # Print summary
    print("\n=== Conversion and Testing Summary ===")
    if not args.skip_clip:
        if clip_model_path:
            print(f"CLIP model: Converted successfully to {clip_model_path}")
            if not args.skip_tests:
                status = "Passed" if test_results.get("clip", False) else "Failed or skipped"
                print(f"CLIP model testing: {status}")
        else:
            print("CLIP model: Conversion failed")
    else:
        print("CLIP model: Skipped")
    
    if not args.skip_clap:
        if clap_model_path:
            print(f"CLAP model: Converted successfully to {clap_model_path}")
            if not args.skip_tests:
                status = "Passed" if test_results.get("clap", False) else "Failed or skipped"
                print(f"CLAP model testing: {status}")
        else:
            print("CLAP model: Conversion failed")
    else:
        print("CLAP model: Skipped")
    
    print("\nProcess completed.")
    
if __name__ == "__main__":
    main()
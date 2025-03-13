import argparse
import subprocess
import sys

from src.classify import FFmpegCommandBuilder

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build FFmpeg/FFplay commands for AI-based video/audio analysis and visualization')
    
    # Input/output options
    parser.add_argument('--input', required=True, help='Input video/audio file')
    parser.add_argument('--output-stats', help='Output statistics file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--skip-confirmation', action='store_true', help='Enable verbose output')

    # Mode selection
    parser.add_argument('--visualization', action='store_true', 
                        help='Use FFplay for visualization instead of FFmpeg for analysis')
    
    # Visualization options
    visualization_group = parser.add_argument_group('Visualization Options')
    visualization_group.add_argument('--box-color', default='red',
                                    help='Color for detection bounding boxes (default: red)')
    visualization_group.add_argument('--text-color', default='yellow',
                                    help='Color for detection text (default: yellow)')
    visualization_group.add_argument('--border-color', 
                                    help='Color for text border (defaults to text color)')
    visualization_group.add_argument('--font-size', type=int, default=20,
                                    help='Font size for detection text (default: 20)')
    
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
    print("\nGenerated command:")
    print(" ".join(cmd))
    
    # Ask user if they want to execute the command
    if args.skip_confirmation:
        response = 'y'
    else:
        response = input("\nDo you want to execute this command? (y/n): ")
    if response.lower() in ['y', 'yes']:
        try:
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            print(f"Error executing command: {e}")
            sys.exit(1)
    else:
        print("Command not executed. You can copy and run it manually.")

if __name__ == "__main__":
    main()
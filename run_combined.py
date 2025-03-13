#!/usr/bin/env python3
import sys
import json
import logging
import os
from pathlib import Path

from src.combine import CombinedAnalyzer

def process_from_existing_files(args):
    """Process analysis using existing files instead of running new analysis.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to saved results JSON file
    """
    tech_json_path = Path(args.from_existing)
    class_txt_path = Path(args.classification_txt)
    
    if not tech_json_path.exists():
        raise FileNotFoundError(f"Technical analysis file not found: {args.from_existing}")
        
    if not class_txt_path.exists():
        raise FileNotFoundError(f"Classification file not found: {args.classification_txt}")
    
    logging.info("Using existing analysis files")
    
    # Load technical analysis
    with open(tech_json_path, 'r') as f:
        technical_data = json.load(f)
    
    # Initialize analyzer with input file from technical data
    analyzer = CombinedAnalyzer(
        input_file=technical_data["input_file"],
        output_dir=args.output
    )
    
    # Manually set the technical data
    analyzer.technical_data = technical_data
    
    # Parse the classification results
    analyzer._parse_classification_results(class_txt_path)
    
    # Combine analyses
    logging.info("Combining analysis results")
    analyzer.combine_analyses()
    
    # Save JSON results
    if args.json:
        json_path = analyzer.save_results(args.json)
    else:
        json_path = analyzer.save_results()
        
    # Print summary
    analyzer.print_summary()
    
    return json_path

def run_new_analysis(args):
    """Run new analysis on an input file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to saved results JSON file
    """
    # Get the current script directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Run technical analysis
    logging.info(f"Running technical analysis on {args.input}")
    output_json = Path(args.output) / f"{Path(args.input).stem}_technical.json" if args.output else Path(f"{Path(args.input).stem}_technical.json")
    
    tech_cmd = [
        sys.executable, str(script_dir / "run_analysis.py"),
        args.input,
        "--scene-threshold", str(args.scene_threshold),
        "--json", str(output_json)
    ]
    
    import subprocess
    try:
        subprocess.run(tech_cmd, check=True)
        logging.info("Technical analysis completed successfully")
        
        # Load the technical data
        with open(output_json, 'r') as f:
            technical_data = json.load(f)
            
        # Run AI classification
        logging.info(f"Running AI classification on {args.input}")
        output_class = Path(args.output) / f"{Path(args.input).stem}_classifications.txt" if args.output else Path(f"{Path(args.input).stem}_classifications.txt")
        
        class_cmd = [
            sys.executable, str(script_dir / "run_classification.py"),
            "--input", args.input,
            "--scene-threshold", str(args.scene_threshold),
            "--temperature", str(args.temperature),
            "--clip-categories", args.clip_categories,
            "--clap-categories", args.clap_categories,
            "--output-stats", str(output_class),
            "--skip-confirmation"
        ]
        
        subprocess.run(class_cmd, check=True)
        logging.info("AI classification completed successfully")
        
        # Initialize analyzer
        analyzer = CombinedAnalyzer(
            input_file=args.input,
            output_dir=args.output
        )
        
        # Set the technical data
        analyzer.technical_data = technical_data
        
        # Parse the classification results
        analyzer._parse_classification_results(output_class)
        
        # Combine analyses
        logging.info("Combining analysis results")
        analyzer.combine_analyses()
        
        # Save JSON results
        if args.json:
            json_path = analyzer.save_results(args.json)
        else:
            json_path = analyzer.save_results()
            
        # Print summary
        analyzer.print_summary()
        
        return json_path
        
    except subprocess.SubprocessError as e:
        logging.error(f"Analysis failed: {e}")
        if hasattr(e, 'stderr'):
            logging.error(f"Error details: {e.stderr}")
        raise

def main():
    """Main entry point for the script."""
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Combined Media Analysis Tool - Integrates technical analysis and AI classifications'
    )
    parser.add_argument('input', help='Path to input media file')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('--scene-threshold', type=float, default=0.3,
                      help='Scene detection threshold (0.0-1.0, lower values detect more scenes)')
    parser.add_argument('--clip-categories', default='resources/labels/clip_combined_analysis.txt',
                      help='Path to CLIP categories file')
    parser.add_argument('--clap-categories', default='resources/labels/clap_combined_analysis.txt',
                      help='Path to CLAP categories file')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='AI classification temperature (lower is more focused)')
    parser.add_argument('--json', help='Path to save JSON analysis results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--from-existing', help='Load from existing technical JSON and skip analysis')
    parser.add_argument('--classification-txt', help='Path to existing classification results text file')
    
    args = parser.parse_args()

    # Create output directory if specified and doesn't exist
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.exists():
            logging.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Determine if using existing files or running new analysis
        if args.from_existing and args.classification_txt:
            json_path = process_from_existing_files(args)
        else:
            json_path = run_new_analysis(args)
            
        logging.info(f"Analysis completed successfully!")
        logging.info(f"JSON results: {json_path}")
        
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
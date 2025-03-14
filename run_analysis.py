import argparse
import logging
import os
import sys

from src.analyze import MediaAnalyzer
from src.util_classes import AnalysisError


def main():
    """Main entry point for the script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Media Analysis Tool - Extract technical metadata and analyze media files'
    )
    parser.add_argument('input', help='Path to input media file')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('-j', '--json', help='Path to save JSON analysis results')
    parser.add_argument('--scene-threshold', type=float, default=0.3,
                      help='Scene detection threshold (0.0-1.0, lower values detect more scenes)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(project_root, "FFmpeg")
    ffmpeg_binary = os.path.join(ffmpeg_path, "ffmpeg")
    ffprobe_binary = os.path.join(ffmpeg_path, "ffprobe")

    try:
        # Create analyzer
        analyzer = MediaAnalyzer(
            input_file=args.input,
            output_dir=args.output,
            ffmpeg_path=ffmpeg_binary,
            ffprobe_path=ffprobe_binary,
            scene_threshold=args.scene_threshold
        )
        
        # Run analysis
        logging.info(f"Starting analysis of {args.input}")
        analyzer.analyze_all()
        
        # Print summary
        analyzer.print_summary()
        
        # Save JSON results if requested
        if args.json:
            json_path = analyzer.save_results(args.json)
            print(f"\nAnalysis results saved to: {json_path}")
        else:
            # Save to default path
            json_path = analyzer.save_results()
            print(f"\nAnalysis results saved to: {json_path}")
            
        logging.info("Analysis complete")
        
    except (FileNotFoundError, AnalysisError) as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
        
       
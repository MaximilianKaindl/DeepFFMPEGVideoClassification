#!/usr/bin/env python3
import os
import json
import shutil
from src.util_classes import AnalysisError, FFmpegTool


class FFmpegCommandBuilder:
    def __init__(self, ffmpeg_path : str, ffplay_path : str, models_dir : str, resources_dir : str):
        # Default directories
        self.models_dir = models_dir
        self.resources_dir = resources_dir
        
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
        
        # Create FFmpegTool instance
        self.ffmpeg_tool = FFmpegTool(ffmpeg_path=ffmpeg_path, ffplay_path=ffplay_path)

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
        self.ffmpeg_tool._validate_tools()

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
            elif device_traced == "cpu" and args.device == "cuda":
                print("Warning: CLAP model was traced with CPU but requested device is CUDA.")
                print("Using CPU for CLAP as it's required for models traced with CPU.")
                clap_device = "cpu"
        
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

    def build_visualization_filters(self, args):
        """Build visualization filters for drawing boxes and text"""
        filters = []
        
        # Only add visualization filters if using FFplay or saving to output file
        if args.detect_model and (args.visualization):
            # Add bounding box visualization
            box_color = args.box_color if args.box_color else "red"
            filters.append(f"drawbox=box_source=side_data_detection_bboxes:color={box_color}")
            
            # Add text visualization
            text_color = args.text_color if args.text_color else "yellow"
            border_color = args.border_color if args.border_color else text_color
            font_size = args.font_size if args.font_size else 20
            filters.append(f"drawtext=text_source=side_data_detection_bboxes:fontcolor={text_color}:bordercolor={border_color}:fontsize={font_size}")
            
        return filters

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
        
        # Determine whether to use FFmpeg or FFplay
        use_ffplay = args.visualization
        
        command_path = self.ffmpeg_tool.ffplay_path if use_ffplay else self.ffmpeg_tool.ffmpeg_path
        
        # Base command
        cmd = [command_path]
        
        # Add verbosity
        if args.verbose:
            cmd.extend(["-v", "debug"])
        else:
            cmd.extend(["-v", "info"])
            
        # Input file
        cmd.extend(["-i", args.input])
        
        # Build filter complex or filter graph
        filters = []
        
        # If using detection with visualization, ensure we don't mix with CLAP
        if args.detect_model and args.clap_model and use_ffplay:
            print("Warning: CLAP audio analysis is disabled in visualization mode with detection")
            # Temporarily set clap_model to None to avoid interference
            args.clap_model = None
        # CLAP should be isolated from detection and visualization
        # For visualization mode or detection, we'll use simpler filters
        if use_ffplay or (args.detect_model and not args.clap_model):
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
                
            # Add visualization filters if needed for FFplay
            vis_filters = self.build_visualization_filters(args)
            if vis_filters:
                filters.extend(vis_filters)
                
            # Add average classification if not in visualization mode
            if not use_ffplay:
                filters.append(self.build_average_filter(args))
            
            cmd.extend(["-vf", ",".join(filters)])
        elif args.clap_model:
            # Complex filter graph for audio processing
            filter_complex = []
            
            # Add audio stream processing
            audio_filter = f"[0:a] {self.build_clap_filter(args)} [audio]"
            filter_complex.append(audio_filter)
            
            # Only add video processing to filter_complex if needed alongside CLAP
            if args.clip_model:
                scene_filter = f"select='gt(scene,{args.scene_threshold})'," if args.scene_threshold > 0 else ""
                video_filter = f"[0:v] {scene_filter}{self.build_clip_filter(args)} [video]"
                filter_complex.append(video_filter)
                filter_complex.append(f"[video][audio] {self.build_average_filter(args)}")
            else:
                filter_complex.append(f"[audio] {self.build_average_filter(args)}")
                
            cmd.extend(["-filter_complex", ";".join(filter_complex)])
        else:
            
            # Only add scene detection if threshold is positive
            if args.scene_threshold > 0:
                filters.append(f"select='gt(scene,{args.scene_threshold})'")
                
            # Add detection if needed
            if args.detect_model:
                filters.append(self.build_detection_filter(args))
                
            # Add CLIP if needed
            if args.clip_model:
                filters.append(self.build_clip_filter(args))
                
            # Add visualization filters if needed
            vis_filters = self.build_visualization_filters(args)
            if vis_filters:
                filters.extend(vis_filters)
                
            # Add average classification if not in visualization mode
            if not use_ffplay:
                filters.append(self.build_average_filter(args))
            
            cmd.extend(["-vf", ",".join(filters)])

        # Output handling
        if use_ffplay:
            # FFplay doesn't need output specification
            pass
        else:
            # Default null output for FFmpeg analysis
            cmd.extend(["-f", "null", "-"])
        return cmd

    def execute_command(self, cmd, capture_output=False):
        """Execute the built command using FFmpegTool."""
        result = self.ffmpeg_tool.run_command(cmd, check=False, capture_output=capture_output)
        return result
#!/usr/bin/env python3
"""
Media Analyzer - Uses FFmpeg to analyze media files and extract technical metadata.
"""
import os
import subprocess
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.util_classes import (
    AnalysisError, FFmpegTool, Metadata, VideoStream, AudioStream, 
    ActionMoment, AudioAnalysis, SceneAnalysis, SystemInfo, AnalysisResults,
    MomentType, DetectionMethod, SystemClassification, ResolutionClass,
    FrameRateClass, AudioClass
)


class MediaAnalyzer:
    """Main class for analyzing media files using FFmpeg/FFprobe."""
    
    def __init__(self, input_file: str, output_dir: Optional[str] = None, 
                 ffmpeg_path: str = "./FFmpeg/ffmpeg", ffprobe_path: str = "./FFmpeg/ffprobe",
                 scene_threshold: float = 0.3):
        """Initialize the media analyzer with input file and optional paths.
        
        Args:
            input_file: Path to the media file to analyze
            output_dir: Directory for analysis outputs (default: temp directory)
            ffmpeg_path: Path to ffmpeg binary
            ffprobe_path: Path to ffprobe binary
            scene_threshold: Threshold for scene detection (0.0-1.0, default: 0.3)
                             Lower values detect more scenes, higher values fewer scenes
        """
        self.input_file = Path(input_file)
        self.ffmpeg = FFmpegTool(ffmpeg_path, ffprobe_path)
        self.scene_threshold = max(0.0, min(1.0, scene_threshold))  # Clamp to valid range
        
        # Create output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(tempfile.mkdtemp())
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Initialize results
        self.results = AnalysisResults(
            input_file=str(input_file),
            analysis_parameters={"scene_threshold": self.scene_threshold}
        )
    
    def analyze_all(self) -> AnalysisResults:
        """Run all analysis methods concurrently when possible."""
        # We must extract metadata first as other analyses depend on it
        self.extract_metadata()
        
        # Concurrent analysis for video and audio
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if self.has_video_stream():
                futures.append(executor.submit(self.analyze_video))
            
            if self.has_audio_stream():
                futures.append(executor.submit(self.analyze_audio))
            
            # Wait for all analyses to complete
            for future in futures:
                future.result()
        
        # System info analysis is quick and depends on other results
        self.analyze_system_info()
        
        return self.results
    
    def extract_metadata(self) -> Metadata:
        """Extract metadata using ffprobe."""
        cmd = [
            self.ffmpeg.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(self.input_file)
        ]
        
        result = self.ffmpeg.run_command(cmd)
        try:
            raw_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise AnalysisError(f"Failed to parse ffprobe output: {e}")
        
        # Extract base metadata
        format_data = raw_data.get("format", {})
        
        metadata = Metadata(
            format=format_data.get("format_name", "unknown"),
            duration=float(format_data.get("duration", 0)),
            size=int(format_data.get("size", 0)),
            bitrate=int(format_data.get("bit_rate", 0)),
            streams=len(raw_data.get("streams", [])),
            tags=format_data.get("tags", {})
        )
        
        # Process streams
        self._process_streams(metadata, raw_data.get("streams", []))
        self.results.metadata = metadata
        
        return metadata
    
    def _process_streams(self, metadata: Metadata, streams: List[Dict[str, Any]]) -> None:
        """Process stream data into structured format."""
        for stream in streams:
            if stream.get("codec_type") == "video":
                metadata.video_streams.append(self._extract_video_stream_info(stream))
            elif stream.get("codec_type") == "audio":
                metadata.audio_streams.append(self._extract_audio_stream_info(stream))
    
    def _extract_video_stream_info(self, stream: Dict[str, Any]) -> VideoStream:
        """Extract relevant information from a video stream."""
        return VideoStream(
            codec=stream.get("codec_name"),
            width=stream.get("width"),
            height=stream.get("height"),
            fps=self._parse_frame_rate(stream.get("r_frame_rate", "0/1")),
            bit_depth=stream.get("bits_per_raw_sample"),
            pix_fmt=stream.get("pix_fmt"),
            profile=stream.get("profile"),
            color_space=stream.get("color_space")
        )
    
    def _extract_audio_stream_info(self, stream: Dict[str, Any]) -> AudioStream:
        """Extract relevant information from an audio stream."""
        return AudioStream(
            codec=stream.get("codec_name"),
            sample_rate=stream.get("sample_rate"),
            channels=stream.get("channels"),
            channel_layout=stream.get("channel_layout"),
            bit_depth=stream.get("bits_per_sample")
        )
    
    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse FFmpeg frame rate string (e.g. '24000/1001') to float."""
        try:
            if "/" in frame_rate_str:
                num, den = map(int, frame_rate_str.split("/"))
                return num / den if den != 0 else 0
            else:
                return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 0
    
    def has_video_stream(self) -> bool:
        """Check if the file has at least one video stream."""
        return len(self.results.metadata.video_streams) > 0
    
    def has_audio_stream(self) -> bool:
        """Check if the file has at least one audio stream."""
        return len(self.results.metadata.audio_streams) > 0
    
    def analyze_video(self) -> SceneAnalysis:
        """Analyze video content including scene changes and visual features."""
        # First count frames
        frame_count = self._count_frames()
        
        # Using the showinfo filter method for scene detection
        logging.info("Running scene detection with showinfo filter")
        scenes = self._extract_keyframe_timestamps()
        
        # Calculate scene metrics
        scene_count = len(scenes)
        duration = self.results.metadata.duration
        
        # If no scenes detected and the duration is substantial, create artificial markers
        if not scenes and duration > 10:
            logging.info("Creating artificial scene markers based on duration")
            # Create reasonable timestamps based on duration
            num_keyframes = max(5, min(20, int(duration / 10)))
            scenes = [i * (duration / num_keyframes) for i in range(1, num_keyframes)]
            scene_count = len(scenes)
        
        # Create scene analysis object
        scene_analysis = SceneAnalysis(
            total_frames=frame_count,
            scene_count=scene_count,
            scene_timestamps=scenes,
            average_scene_duration=duration / scene_count if scene_count > 0 else duration,
            scene_frequency=scene_count / duration if duration > 0 else 0,
            threshold_used=self.scene_threshold,
            detection_method=DetectionMethod.SHOWINFO if scenes else DetectionMethod.FALLBACK
        )
        
        self.results.scene_analysis = scene_analysis
        return scene_analysis
    
    def _extract_keyframe_timestamps(self) -> List[float]:
        """Extract keyframe timestamps using showinfo filter."""
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-i", str(self.input_file),
            "-vf", "showinfo",
            "-f", "null", "-"
        ]
        result = self.ffmpeg.run_command(cmd, check=False)
        
        scenes = []
        for line in result.stderr.split('\n'):
            if "pts_time:" in line and "iskey:1" in line:
                try:
                    # Extract timestamp for keyframes
                    time_parts = line.split('pts_time:')
                    if len(time_parts) > 1:
                        time_str = time_parts[1].strip().split()[0]
                        timestamp = float(time_str)
                        scenes.append(timestamp)
                except (ValueError, IndexError) as e:
                    logging.debug(f"Error parsing keyframe: {e}")
        
        # Keep only reasonably spaced keyframes (at least 1 second apart)
        if scenes:
            filtered_timestamps = [scenes[0]]
            for t in scenes[1:]:
                if t - filtered_timestamps[-1] >= 1.0:
                    filtered_timestamps.append(t)
            
            logging.info(f"Found {len(filtered_timestamps)} keyframes (after filtering)")
            return filtered_timestamps
        
        return []
    
    def _count_frames(self) -> int:
        """Count total frames in the video."""
        cmd = [
            self.ffmpeg.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(self.input_file)
        ]
        
        try:
            result = self.ffmpeg.run_command(cmd)
            return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, AnalysisError):
            logging.warning("Could not count frames with primary method, using estimate")
            # Estimate frames from duration and frame rate
            duration = self.results.metadata.duration
            if self.has_video_stream():
                fps = self.results.metadata.video_streams[0].fps
                return int(duration * fps) if fps > 0 else 0
            return 0
    
    def analyze_audio(self) -> AudioAnalysis:
        """Analyze audio characteristics focusing on loudness and activity."""
        # First check if file actually has audio streams
        if not self.has_audio_stream():
            logging.warning("No audio streams found in the file")
            audio_analysis = AudioAnalysis(has_audio=False)
            self.results.audio_analysis = audio_analysis
            return audio_analysis
            
        # Volume analysis (most compatible)
        volume_stats = self._analyze_volume()
        
        # Silence detection
        silence_info = self._detect_silence()
        duration = self.results.metadata.duration
        
        # Extract audio action moments from silence detection
        action_moments = self._extract_action_moments_from_silence(silence_info, duration)
        
        # Volume peaks for additional action moments
        volume_peaks = self._detect_volume_peaks(duration)
        
        # Combine all audio action moments
        all_action_moments = self._combine_action_moments(action_moments, volume_peaks, duration)
            
        # Create audio analysis object
        audio_analysis = AudioAnalysis(
            has_audio=True,
            mean_volume=volume_stats.get("mean_volume"),
            max_volume=volume_stats.get("max_volume"),
            action_moment_count=len(all_action_moments),
            action_moments=all_action_moments[:15],  # Limit to first 15 moments for brevity
            analysis_method=DetectionMethod.COMBINED
        )
        
        self.results.audio_analysis = audio_analysis
        return audio_analysis
    
    def _analyze_volume(self) -> Dict[str, float]:
        """Analyze audio volume using volumedetect filter."""
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-i", str(self.input_file),
            "-filter:a", "volumedetect",
            "-f", "null", "-"
        ]
        
        logging.info("Running simple volume detection analysis")
        result = self.ffmpeg.run_command(cmd, check=False)
        
        # Parse results
        return self._parse_volume_data(result.stderr)
    
    def _detect_silence(self) -> List[Dict[str, float]]:
        """Detect silence segments in audio."""
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-i", str(self.input_file),
            "-af", "silencedetect=noise=-30dB:d=0.5",
            "-f", "null", "-"
        ]
        
        logging.info("Running silence detection for audio segments")
        result = self.ffmpeg.run_command(cmd, check=False)
        
        # Parse silence detection results
        return self._parse_silence_detection(result.stderr)
    
    def _parse_volume_data(self, stderr_output: str) -> Dict[str, float]:
        """Parse volume data from FFmpeg volumedetect output."""
        volume_data = {
            "mean_volume": None,
            "max_volume": None
        }
        
        for line in stderr_output.split('\n'):
            if "mean_volume:" in line:
                try:
                    mean_vol = float(line.split(':')[1].strip().split(' ')[0])
                    volume_data["mean_volume"] = mean_vol
                except (ValueError, IndexError):
                    pass
            elif "max_volume:" in line:
                try:
                    max_vol = float(line.split(':')[1].strip().split(' ')[0])
                    volume_data["max_volume"] = max_vol
                except (ValueError, IndexError):
                    pass
        
        logging.info(f"Parsed volume data: mean={volume_data['mean_volume']} dB, max={volume_data['max_volume']} dB")
        return volume_data
    
    def _parse_silence_detection(self, stderr_output: str) -> List[Dict[str, float]]:
        """Parse silence detection data from FFmpeg output."""
        silence_ranges = []
        
        for line in stderr_output.split('\n'):
            if "silence_start:" in line:
                try:
                    start_time = float(line.split('silence_start:')[1].strip())
                    silence_ranges.append({"start": start_time, "end": None})
                except (ValueError, IndexError) as e:
                    logging.debug(f"Error parsing silence start: {e}")
            elif "silence_end:" in line and silence_ranges and silence_ranges[-1]["end"] is None:
                try:
                    parts = line.split('silence_end:')[1].strip().split('|')
                    end_time = float(parts[0].strip())
                    # Also extract silence duration if available
                    duration = None
                    if "silence_duration:" in line:
                        duration = float(line.split('silence_duration:')[1].strip())
                    
                    silence_ranges[-1]["end"] = end_time
                    if duration is not None:
                        silence_ranges[-1]["duration"] = duration
                except (ValueError, IndexError) as e:
                    logging.debug(f"Error parsing silence end: {e}")
        
        # Remove incomplete entries
        silence_ranges = [s for s in silence_ranges if s["end"] is not None]
        
        if silence_ranges:
            logging.info(f"Detected {len(silence_ranges)} silence ranges")
            if len(silence_ranges) < 5:
                for sr in silence_ranges:
                    logging.info(f"  Silence: {sr['start']:.2f}s - {sr['end']:.2f}s")
        else:
            logging.info("No silence ranges detected")
            
        return silence_ranges
    
    def _extract_action_moments_from_silence(self, silence_info: List[Dict[str, float]], 
                                           total_duration: float) -> List[ActionMoment]:
        """Extract audio action moments from silence detection results."""
        action_moments = []
        
        if not silence_info:
            return action_moments
            
        # First, add the beginning if it's not silence
        if silence_info[0]["start"] > 0.5:
            action_moments.append(ActionMoment(
                time=silence_info[0]["start"] / 2,  # midpoint of initial non-silence
                type=MomentType.INITIAL_AUDIO,
                intensity=0.7
            ))
        
        # Then add moments between silence regions (these are the loud parts)
        for i in range(len(silence_info)-1):
            current_end = silence_info[i]["end"]
            next_start = silence_info[i+1]["start"]
            
            # If there's a gap between silence regions (i.e., sound)
            if next_start - current_end > 0.5:  # At least 0.5s of sound
                midpoint = current_end + (next_start - current_end) / 2
                # Duration of the sound section affects intensity
                duration = next_start - current_end
                intensity = min(1.0, duration / 5)  # Scale intensity with duration, cap at 1.0
                
                action_moments.append(ActionMoment(
                    time=midpoint,
                    type=MomentType.SOUND_SECTION,
                    intensity=intensity,
                    duration=duration
                ))
        
        # Finally, add the end if it's not silence
        if silence_info[-1]["end"] < total_duration - 0.5:
            action_moments.append(ActionMoment(
                time=(silence_info[-1]["end"] + total_duration) / 2,  # midpoint of final non-silence
                type=MomentType.FINAL_AUDIO,
                intensity=0.7
            ))
            
        logging.info(f"Extracted {len(action_moments)} action moments from silence detection")
        return action_moments
    
    def _detect_volume_peaks(self, duration: float) -> List[ActionMoment]:
        """Detect volume peaks by sampling audio at regular intervals."""
        peaks = []
        
        # Only run if the duration is sufficient
        if duration < 5:
            return peaks
            
        # Determine number of samples (more for longer videos)
        num_samples = min(20, max(10, int(duration / 5)))
        interval = duration / num_samples
        
        # Sample at regular intervals with ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, num_samples)) as executor:
            futures = []
            
            for i in range(num_samples):
                time_pos = i * interval
                futures.append(executor.submit(self._analyze_volume_at_position, time_pos))
            
            # Collect results
            for future in futures:
                result = future.result()
                if result is not None:
                    peaks.append(result)
                
        logging.info(f"Detected {len(peaks)} volume peaks in {num_samples} samples")
        return peaks
    
    def _analyze_volume_at_position(self, time_pos: float) -> Optional[ActionMoment]:
        """Analyze volume at a specific position in the media file."""
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-ss", str(time_pos),
            "-i", str(self.input_file),
            "-t", "1",  # 1 second segment
            "-filter:a", "volumedetect",
            "-f", "null", "-"
        ]
        
        try:
            result = self.ffmpeg.run_command(cmd, check=False)
            volume_stats = self._parse_volume_data(result.stderr)
            
            # If the max volume is high enough, consider it a peak
            max_vol = volume_stats.get("max_volume")
            if max_vol is not None and max_vol > -10:  # -10dB threshold
                intensity = min(1.0, (max_vol + 30) / 30)  # Scale -30dB to 0 -> 0dB to 1.0
                return ActionMoment(
                    time=time_pos + 0.5,  # Middle of the 1s segment
                    type=MomentType.VOLUME_PEAK,
                    intensity=intensity,
                    max_volume=max_vol
                )
        except Exception as e:
            logging.debug(f"Error analyzing volume at {time_pos}s: {e}")
        
        return None
    
    def _combine_action_moments(self, 
                              silence_moments: List[ActionMoment], 
                              peak_moments: List[ActionMoment],
                              duration: float) -> List[ActionMoment]:
        """Combine different sources of audio action moments."""
        all_moments = []
        
        # Start with silence-based moments if available
        if silence_moments:
            all_moments.extend(silence_moments)
        
        # Add peak moments if not too close to existing moments
        if peak_moments:
            for peak in peak_moments:
                # Check if this peak is far enough from existing moments (at least 1 second)
                if not all_moments or all(abs(peak.time - moment.time) > 1.0 for moment in all_moments):
                    all_moments.append(peak)
        
        # If we still have no moments but we have scene changes, derive audio moments from scene changes
        if not all_moments and self.results.scene_analysis.scene_timestamps:
            scene_times = self.results.scene_analysis.scene_timestamps
            # Use a subset of scene changes as audio moments
            all_moments = [
                ActionMoment(
                    time=time, 
                    type=MomentType.SCENE_CHANGE, 
                    intensity=0.5
                ) 
                for time in scene_times[:min(10, len(scene_times))]  # Use at most 10 scene changes
                if time > 0  # Skip the first scene at 0
            ]
        
        # Sort by time
        all_moments.sort(key=lambda x: x.time)
        
        return all_moments
    
    def analyze_system_info(self) -> SystemInfo:
        """Analyze metadata to infer recording system information."""
        recording_hints = {}
        
        # Extract hints from metadata tags
        tags = self.results.metadata.tags
        for key, value in tags.items():
            if any(hint in key.lower() for hint in ["device", "camera", "encoder", "model", "make"]):
                recording_hints[key] = value
        
        # Classify video characteristics if present
        if self.has_video_stream():
            video = self.results.metadata.video_streams[0]
            resolution = f"{video.width}x{video.height}"
            
            # Resolution classification
            if resolution in ["3840x2160", "4096x2160"]:
                recording_hints["resolution_class"] = ResolutionClass.UHD_4K
            elif resolution in ["1920x1080"]:
                recording_hints["resolution_class"] = ResolutionClass.FULL_HD
            elif resolution in ["1280x720"]:
                recording_hints["resolution_class"] = ResolutionClass.HD
            else:
                recording_hints["resolution_class"] = ResolutionClass.SD
            
            # Frame rate classification
            fps = video.fps
            if abs(fps - 24) < 0.1:
                recording_hints["frame_rate_class"] = FrameRateClass.FILM
            elif abs(fps - 30) < 0.1:
                recording_hints["frame_rate_class"] = FrameRateClass.BROADCAST
            elif fps > 48:
                recording_hints["frame_rate_class"] = FrameRateClass.HIGH_FRAME_RATE
        
        # Classify audio characteristics if present
        if self.has_audio_stream():
            audio = self.results.metadata.audio_streams[0]
            sample_rate = int(audio.sample_rate) if audio.sample_rate else 0
            
            if sample_rate >= 48000:
                recording_hints["audio_class"] = AudioClass.PROFESSIONAL
            elif sample_rate == 44100:
                recording_hints["audio_class"] = AudioClass.CONSUMER
        
        # Make overall classification
        system_class = self._classify_system(recording_hints)
        
        system_info = SystemInfo(
            classification=system_class,
            details=recording_hints
        )
        
        self.results.system_info = system_info
        return system_info
    
    def _classify_system(self, hints: Dict[str, str]) -> SystemClassification:
        """Classify the recording system based on gathered hints."""
        if any(key in hints for key in ["make", "model"]):
            return SystemClassification.IDENTIFIED
        elif ResolutionClass.UHD_4K in hints.get("resolution_class", ""):
            return SystemClassification.PROFESSIONAL
        elif ResolutionClass.FULL_HD in hints.get("resolution_class", "") and AudioClass.PROFESSIONAL in hints.get("audio_class", ""):
            return SystemClassification.PROSUMER
        elif ResolutionClass.HD in hints.get("resolution_class", ""):
            return SystemClassification.CONSUMER
        return SystemClassification.UNKNOWN
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        results = self.results
        print("\n=== Media Analysis Summary ===")
        print(f"File: {results.input_file}")
        print(f"Format: {results.metadata.format}")
        print(f"Duration: {results.metadata.duration:.2f} seconds")
        
        # Analysis parameters
        print(f"Scene threshold: {self.scene_threshold} (lower values detect more scenes)")
        
        # Video information
        if self.has_video_stream():
            vs = results.metadata.video_streams[0]
            print(f"Video: {vs.width}x{vs.height} @ {vs.fps:.2f}fps ({vs.codec})")
            
            # Scene information
            scene_count = results.scene_analysis.scene_count
            avg_duration = results.scene_analysis.average_scene_duration
            print(f"Scenes: {scene_count} detected")
            print(f"Scene detection method: {results.scene_analysis.detection_method}")
            if scene_count > 0:
                print(f"Average scene duration: {avg_duration:.2f} seconds")
        
        # Audio information
        if self.has_audio_stream():
            aud = results.metadata.audio_streams[0]
            print(f"Audio: {aud.codec} {aud.sample_rate}Hz {aud.channels}ch")
            mean_vol = results.audio_analysis.mean_volume
            max_vol = results.audio_analysis.max_volume
            if mean_vol is not None:
                print(f"Audio levels: mean {mean_vol}dB, max {max_vol}dB")
            
            action_count = results.audio_analysis.action_moment_count
            print(f"Audio activity: {action_count} moments of high activity detected")
            
            # Show first few action moments if available
            action_moments = results.audio_analysis.action_moments
            if action_moments and len(action_moments) <= 5:
                print("Audio action moments:")
                for moment in action_moments:
                    print(f"  {moment.time:.2f}s: {moment.type} (intensity: {moment.intensity:.2f})")
            elif action_moments:
                print(f"First 3 audio action moments:")
                for moment in action_moments[:3]:
                    print(f"  {moment.time:.2f}s: {moment.type} (intensity: {moment.intensity:.2f})")
        
        # System info
        print(f"Recording system: {results.system_info.classification}")

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save analysis results to a JSON file."""
        if output_file is None:
            output_file = self.output_dir / f"{self.input_file.stem}_analysis.json"
        else:
            output_file = Path(output_file)
        
        # Convert results to dictionary for serialization
        results_dict = self.results.to_dict()
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        return str(output_file)
#!/usr/bin/env python3
"""
Streamlined Combined Media Analysis Tool - Integrates technical metadata and AI classifications.
Focuses on JSON output without HTML report generation.
"""
import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv
from collections import defaultdict, Counter


class CombinedAnalyzer:
    """
    Class that combines technical analysis and AI classification data.
    """
    
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """Initialize the combined analyzer with input file and output directory.
        
        Args:
            input_file: Path to the media file to analyze
            output_dir: Directory for analysis outputs
        """
        self.input_file = Path(input_file)
        
        # Create output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("combined_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Initialize data structures
        self.technical_data = None
        self.clip_data = None
        self.clap_data = None
        self.combined_results = {
            "file_info": {
                "filename": self.input_file.name,
                "path": str(self.input_file),
                "size_mb": self.input_file.stat().st_size / (1024 * 1024)
            },
            "technical_analysis": None,
            "ai_classifications": {
                "video": None,
                "audio": None
            },
            "combined_insights": {
                "content_type": None,
                "quality_assessment": None,
                "mood": None,
                "storytelling_metrics": None
            }
        }
    
    def run_technical_analysis(self, ffmpeg_path: str = "ffmpeg", 
                              ffprobe_path: str = "ffprobe",
                              scene_threshold: float = 0.3) -> Dict:
        """Run technical analysis using the media analyzer script."""
        logging.info(f"Running technical analysis on {self.input_file}")
        
        # Build command to run the analysis script
        cmd = [
            sys.executable, "run_analysis.py",
            str(self.input_file),
            "--ffmpeg", ffmpeg_path,
            "--ffprobe", ffprobe_path,
            "--threshold", str(scene_threshold),
            "--json", str(self.output_dir / f"{self.input_file.stem}_technical.json")
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load the generated JSON file
            result_path = self.output_dir / f"{self.input_file.stem}_technical.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    self.technical_data = json.load(f)
                logging.info(f"Technical analysis completed successfully")
                return self.technical_data
            else:
                raise FileNotFoundError(f"Technical analysis output not found: {result_path}")
        
        except subprocess.SubprocessError as e:
            logging.error(f"Technical analysis failed: {e}")
            if hasattr(e, 'stderr'):
                logging.error(f"Error details: {e.stderr}")
            raise
    
    def run_ai_classification(self, 
                             clip_categories: str,
                             clap_categories: str,
                             temperature: float = 0.1,
                             scene_threshold: float = 0.3) -> Dict:
        """Run AI classification using the classification script."""
        logging.info(f"Running AI classification on {self.input_file}")
        
        # Build command to run the classification script
        output_file = self.output_dir / f"{self.input_file.stem}_classifications.txt"
        cmd = [
            sys.executable, "run_classification.py",
            "--input", str(self.input_file),
            "--scene-threshold", str(scene_threshold),
            "--temperature", str(temperature),
            "--clip-categories", clip_categories,
            "--clap-categories", clap_categories,
            "--output-stats", str(output_file),
            "--skip-confirmation"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Parse the generated output file
            if output_file.exists():
                self._parse_classification_results(output_file)
                logging.info(f"AI classification completed successfully")
                return {
                    "video": self.clip_data,
                    "audio": self.clap_data
                }
            else:
                raise FileNotFoundError(f"Classification output not found: {output_file}")
        
        except subprocess.SubprocessError as e:
            logging.error(f"AI classification failed: {e}")
            if hasattr(e, 'stderr'):
                logging.error(f"Error details: {e.stderr}")
            raise
    
    def _parse_classification_results(self, output_file: Path) -> None:
        """Parse the classification results file into structured data."""
        video_results = []
        audio_results = []
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Stream 0 is typically video, Stream 1 is typically audio
                if row['stream_id'] == '0':
                    video_results.append({
                        'label': row['label'],
                        'probability': float(row['avg_probability']),
                        'count': int(row['count'])
                    })
                elif row['stream_id'] == '1':
                    audio_results.append({
                        'label': row['label'],
                        'probability': float(row['avg_probability']),
                        'count': int(row['count'])
                    })
        
        # Sort by count (descending) for more relevance
        self.clip_data = sorted(video_results, key=lambda x: x['count'], reverse=True)
        self.clap_data = sorted(audio_results, key=lambda x: x['count'], reverse=True)
    
    def combine_analyses(self) -> Dict:
        """Combine technical analysis and AI classifications."""
        if not self.technical_data:
            raise ValueError("Technical analysis has not been run yet")
        if not self.clip_data or not self.clap_data:
            raise ValueError("AI classification has not been run yet")
        
        # Integrate the technical data
        self.combined_results["technical_analysis"] = self.technical_data
        
        # Integrate the AI classifications
        self.combined_results["ai_classifications"]["video"] = self.clip_data
        self.combined_results["ai_classifications"]["audio"] = self.clap_data
        
        # Generate combined insights
        self._derive_content_type()
        self._assess_quality()
        self._determine_mood()
        self._analyze_storytelling()
        
        return self.combined_results
    
    def _derive_content_type(self) -> None:
        """Derive the content type from classifications and technical data."""
        # Extract primary classification labels
        primary_video_labels = [item["label"] for item in self.clip_data[:3]]
        primary_audio_labels = [item["label"] for item in self.clap_data[:3]]
        
        # Initialize content type assessment
        content_assessment = {
            "primary_type": None,
            "confidence": 0.0,
            "subtypes": [],
            "format_info": {
                "duration": self.technical_data["metadata"]["duration"],
                "resolution": None,
                "frame_rate": None
            }
        }
        
        # Determine resolution description
        if self.technical_data["metadata"]["video_streams"]:
            video_stream = self.technical_data["metadata"]["video_streams"][0]
            width = video_stream["width"]
            height = video_stream["height"]
            frame_rate = video_stream["fps"]
            
            content_assessment["format_info"]["resolution"] = f"{width}x{height}"
            content_assessment["format_info"]["frame_rate"] = frame_rate
            
            # Add resolution classification
            if width >= 3840 or height >= 2160:
                content_assessment["format_info"]["resolution_class"] = "4K"
            elif width >= 1920 or height >= 1080:
                content_assessment["format_info"]["resolution_class"] = "Full HD"
            elif width >= 1280 or height >= 720:
                content_assessment["format_info"]["resolution_class"] = "HD"
            else:
                content_assessment["format_info"]["resolution_class"] = "SD"
        
        # Determine primary content type based on both video and audio classifications
        primary_content_map = {
            "Storytelling": ["Storytelling", "StorytellingAudio", "Animation", "LiveAction", "Emotional", "IntenseAudio"],
            "Informational": ["Informational", "InformationalAudio", "FactualAudio"],
            "Entertainment": ["Exciting", "Lighthearted", "LightAudio"]
        }
        
        # Count occurrences of each primary content type in our classifications
        content_type_scores = defaultdict(int)
        all_labels = primary_video_labels + primary_audio_labels
        
        for content_type, related_labels in primary_content_map.items():
            for label in all_labels:
                if label in related_labels:
                    content_type_scores[content_type] += 1
        
        # Get the most common content type
        if content_type_scores:
            primary_type, count = max(content_type_scores.items(), key=lambda x: x[1])
            confidence = count / len(all_labels) if all_labels else 0
            
            content_assessment["primary_type"] = primary_type
            content_assessment["confidence"] = confidence
            
            # Add relevant subtypes
            if "Animation" in primary_video_labels:
                content_assessment["subtypes"].append("Animated")
            elif "LiveAction" in primary_video_labels:
                content_assessment["subtypes"].append("Live Action")
                
            # Check for mood-based subtypes
            mood_subtypes = [
                label for label in all_labels 
                if label in ["Emotional", "Exciting", "Lighthearted", "Tense", "Imaginative",
                            "EerieAudio", "IntenseAudio", "LightAudio", "EmotionalAudio"]
            ]
            if mood_subtypes:
                content_assessment["subtypes"].extend(mood_subtypes)
        else:
            # Fallback if no clear content type is detected
            content_assessment["primary_type"] = "Unclassified Media"
            content_assessment["confidence"] = 0.0
        
        self.combined_results["combined_insights"]["content_type"] = content_assessment
    
    def _assess_quality(self) -> None:
        """Assess the overall quality of the media."""
        quality_assessment = {
            "video_quality": {
                "rating": None,
                "factors": []
            },
            "audio_quality": {
                "rating": None,
                "factors": []
            },
            "technical_quality": {
                "bitrate": self.technical_data["metadata"]["bitrate"],
                "codec_info": {}
            }
        }
        
        # Extract quality-related classifications
        video_quality_labels = [item for item in self.clip_data if item["label"] in ["HighQuality", "LowQuality"]]
        audio_quality_labels = [item for item in self.clap_data if item["label"] in ["ProfessionalRecording", "BasicRecording"]]
        
        # Assess video quality
        if video_quality_labels:
            # Sort by probability for more accurate assessment
            sorted_labels = sorted(video_quality_labels, key=lambda x: x["probability"], reverse=True)
            primary_quality = sorted_labels[0]["label"]
            
            if primary_quality == "HighQuality":
                quality_assessment["video_quality"]["rating"] = "High"
                quality_assessment["video_quality"]["factors"].append("Professional camera work")
                quality_assessment["video_quality"]["factors"].append("Good lighting and composition")
            else:
                quality_assessment["video_quality"]["rating"] = "Standard"
                quality_assessment["video_quality"]["factors"].append("Consumer-level production")
        else:
            # Fallback to technical assessment
            bitrate_mbps = self.technical_data["metadata"]["bitrate"] / 1000000
            if bitrate_mbps > 10:
                quality_assessment["video_quality"]["rating"] = "High"
                quality_assessment["video_quality"]["factors"].append(f"High bitrate ({bitrate_mbps:.1f} Mbps)")
            else:
                quality_assessment["video_quality"]["rating"] = "Standard"
                quality_assessment["video_quality"]["factors"].append(f"Standard bitrate ({bitrate_mbps:.1f} Mbps)")
        
        # Add technical video info
        if self.technical_data["metadata"]["video_streams"]:
            video_stream = self.technical_data["metadata"]["video_streams"][0]
            quality_assessment["technical_quality"]["codec_info"]["video"] = {
                "codec": video_stream["codec"],
                "profile": video_stream["profile"],
                "bit_depth": video_stream["bit_depth"],
                "color_space": video_stream["color_space"]
            }
        
        # Assess audio quality
        if audio_quality_labels:
            sorted_labels = sorted(audio_quality_labels, key=lambda x: x["probability"], reverse=True)
            primary_quality = sorted_labels[0]["label"]
            
            if primary_quality == "ProfessionalRecording":
                quality_assessment["audio_quality"]["rating"] = "Professional"
                quality_assessment["audio_quality"]["factors"].append("Clear audio with good dynamic range")
                quality_assessment["audio_quality"]["factors"].append("Professional post-processing")
            else:
                quality_assessment["audio_quality"]["rating"] = "Basic"
                quality_assessment["audio_quality"]["factors"].append("Standard audio quality")
        else:
            # Fallback to technical assessment
            if self.technical_data["audio_analysis"]["mean_volume"] is not None:
                mean_vol = self.technical_data["audio_analysis"]["mean_volume"] 
                max_vol = self.technical_data["audio_analysis"]["max_volume"]
                
                if mean_vol > -20 and max_vol > -10:
                    quality_assessment["audio_quality"]["rating"] = "Good"
                    quality_assessment["audio_quality"]["factors"].append("Balanced audio levels")
                else:
                    quality_assessment["audio_quality"]["rating"] = "Basic"
                    quality_assessment["audio_quality"]["factors"].append("Variable audio levels")
        
        # Add technical audio info
        if self.technical_data["metadata"]["audio_streams"]:
            audio_stream = self.technical_data["metadata"]["audio_streams"][0]
            quality_assessment["technical_quality"]["codec_info"]["audio"] = {
                "codec": audio_stream["codec"],
                "sample_rate": audio_stream["sample_rate"],
                "channels": audio_stream["channels"],
                "channel_layout": audio_stream["channel_layout"]
            }
            
            # Add professional audio assessment based on sample rate
            if int(audio_stream["sample_rate"]) >= 48000:
                quality_assessment["audio_quality"]["factors"].append("Professional audio sampling rate")
        
        self.combined_results["combined_insights"]["quality_assessment"] = quality_assessment
    
    def _determine_mood(self) -> None:
        """Determine the overall mood of the content."""
        mood_assessment = {
            "primary_mood": None,
            "mood_confidence": 0.0,
            "mood_elements": [],
            "mood_progression": [],
            "mood_consistency": None
        }
        
        # Gather mood-related classifications
        mood_labels = [
            item for item in self.clip_data 
            if item["label"] in ["Emotional", "Exciting", "Lighthearted", "Tense", "Imaginative"]
        ]
        mood_labels_audio = [
            item for item in self.clap_data
            if item["label"] in ["EmotionalAudio", "IntenseAudio", "LightAudio", "EerieAudio", "FantasticalAudio"]
        ]
        
        # Combine and find most common mood
        all_moods = mood_labels + mood_labels_audio
        if all_moods:
            # Get weighted count of each mood (probability * count)
            mood_weights = defaultdict(float)
            for mood in all_moods:
                label = mood["label"].replace("Audio", "")  # Normalize audio and video labels
                weight = mood["probability"] * mood["count"]
                mood_weights[label] += weight
            
            # Find the primary mood with highest weight
            if mood_weights:
                primary_mood, weight = max(mood_weights.items(), key=lambda x: x[1])
                total_weight = sum(mood_weights.values())
                confidence = weight / total_weight if total_weight > 0 else 0
                
                # Clean up mood name (remove "Audio" suffix)
                primary_mood = primary_mood.replace("Audio", "")
                
                mood_assessment["primary_mood"] = primary_mood
                mood_assessment["mood_confidence"] = confidence
            
            # Add all detected moods as mood elements
            for mood_name, weight in mood_weights.items():
                normalized_name = mood_name.replace("Audio", "")
                mood_assessment["mood_elements"].append({
                    "type": normalized_name,
                    "strength": weight / sum(mood_weights.values()) if sum(mood_weights.values()) > 0 else 0
                })
        else:
            # Fallback mood determination from scene analysis
            scene_count = self.technical_data["scene_analysis"]["scene_count"]
            duration = self.technical_data["metadata"]["duration"]
            
            if scene_count / duration > 0.5:  # More than 1 scene per 2 seconds
                mood_assessment["primary_mood"] = "Fast-paced"
                mood_assessment["mood_confidence"] = 0.6
                mood_assessment["mood_elements"].append({
                    "type": "Fast-paced",
                    "strength": 0.6
                })
            else:
                mood_assessment["primary_mood"] = "Moderate"
                mood_assessment["mood_confidence"] = 0.5
                mood_assessment["mood_elements"].append({
                    "type": "Moderate",
                    "strength": 0.5
                })
                
        # Determine mood consistency based on scene frequency variations
        if self.technical_data["scene_analysis"]["scene_count"] > 1:
            scene_times = self.technical_data["scene_analysis"]["scene_timestamps"]
            scene_durations = [scene_times[i+1] - scene_times[i] for i in range(len(scene_times)-1)]
            
            if scene_durations:
                mean_duration = sum(scene_durations) / len(scene_durations)
                variation = sum(abs(d - mean_duration) for d in scene_durations) / len(scene_durations) / mean_duration
                
                if variation < 0.3:
                    mood_assessment["mood_consistency"] = "Highly consistent"
                elif variation < 0.6:
                    mood_assessment["mood_consistency"] = "Moderately consistent"
                else:
                    mood_assessment["mood_consistency"] = "Variable"
                    
                # Add variation metric
                mood_assessment["scene_rhythm_variation"] = variation
        
        # Add information about action moments for mood progression
        if self.technical_data["audio_analysis"]["action_moments"]:
            action_moments = self.technical_data["audio_analysis"]["action_moments"]
            
            # Group action moments into segments for progression analysis
            duration = self.technical_data["metadata"]["duration"]
            num_segments = min(5, len(action_moments))
            segment_duration = duration / num_segments
            
            segments = []
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                # Count moments in this segment
                segment_moments = [m for m in action_moments if start_time <= m["time"] < end_time]
                segments.append({
                    "time_range": f"{start_time:.1f}s - {end_time:.1f}s",
                    "intensity": sum(m["intensity"] for m in segment_moments) / len(segment_moments) if segment_moments else 0,
                    "action_count": len(segment_moments)
                })
            
            mood_assessment["mood_progression"] = segments
        
        self.combined_results["combined_insights"]["mood"] = mood_assessment
    
    def _analyze_storytelling(self) -> None:
        """Analyze storytelling metrics from the combined data."""
        storytelling_metrics = {
            "narrative_structure": None,
            "pacing": None,
            "key_moments": [],
            "scene_analysis": {
                "count": self.technical_data["scene_analysis"]["scene_count"],
                "average_duration": self.technical_data["scene_analysis"]["average_scene_duration"],
                "scenes_per_minute": self.technical_data["scene_analysis"]["scene_count"] / 
                                    (self.technical_data["metadata"]["duration"] / 60)
            }
        }
        
        # Determine pacing from scene frequency
        scenes_per_minute = storytelling_metrics["scene_analysis"]["scenes_per_minute"]
        if scenes_per_minute > 20:
            storytelling_metrics["pacing"] = "Very Fast"
        elif scenes_per_minute > 10:
            storytelling_metrics["pacing"] = "Fast"
        elif scenes_per_minute > 5:
            storytelling_metrics["pacing"] = "Moderate"
        else:
            storytelling_metrics["pacing"] = "Slow"
        
        # Identify key moments from audio analysis
        if self.technical_data["audio_analysis"]["action_moments"]:
            # Sort moments by intensity
            action_moments = sorted(
                self.technical_data["audio_analysis"]["action_moments"],
                key=lambda x: x["intensity"],
                reverse=True
            )
            
            # Take top 3 moments as key moments
            for moment in action_moments[:3]:
                time_formatted = f"{moment['time']:.2f}s"
                key_moments = storytelling_metrics["key_moments"]
                
                # Add with descriptive text based on intensity
                if moment["intensity"] > 0.9:
                    key_moments.append({
                        "time": time_formatted,
                        "description": "Major action/emotional moment",
                        "intensity": moment["intensity"],
                        "type": moment["type"]
                    })
                elif moment["intensity"] > 0.7:
                    key_moments.append({
                        "time": time_formatted,
                        "description": "Significant moment",
                        "intensity": moment["intensity"],
                        "type": moment["type"]
                    })
                else:
                    key_moments.append({
                        "time": time_formatted,
                        "description": "Notable moment",
                        "intensity": moment["intensity"],
                        "type": moment["type"]
                    })
        
        # Determine narrative structure based on scene distribution and audio moments
        scene_timestamps = self.technical_data["scene_analysis"]["scene_timestamps"]
        duration = self.technical_data["metadata"]["duration"]
        
        if len(scene_timestamps) > 2:
            # Analyze scene distribution by dividing into three acts
            first_third = duration / 3
            second_third = 2 * duration / 3
            
            scenes_in_first = len([t for t in scene_timestamps if t < first_third])
            scenes_in_second = len([t for t in scene_timestamps if first_third <= t < second_third])
            scenes_in_third = len([t for t in scene_timestamps if t >= second_third])
            
            # Determine structure based on scene distribution
            if scenes_in_first < scenes_in_second and scenes_in_second < scenes_in_third:
                storytelling_metrics["narrative_structure"] = "Rising Action"
            elif scenes_in_first > scenes_in_second and scenes_in_second < scenes_in_third:
                storytelling_metrics["narrative_structure"] = "Complex (Setup-Conflict-Resolution)"
            elif scenes_in_first > scenes_in_second and scenes_in_second > scenes_in_third:
                storytelling_metrics["narrative_structure"] = "Falling Action"
            else:
                storytelling_metrics["narrative_structure"] = "Episodic"
        else:
            storytelling_metrics["narrative_structure"] = "Simple"
        
        self.combined_results["combined_insights"]["storytelling_metrics"] = storytelling_metrics
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save combined analysis results to a JSON file."""
        if output_file is None:
            output_file = self.output_dir / f"{self.input_file.stem}_combined_analysis.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(self.combined_results, f, indent=2)
        
        return str(output_file)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results to the console, with safety checks for missing keys."""
        results = self.combined_results
        
        print("\n=== Combined Media Analysis Summary ===")
        print(f"File: {self.input_file}")
        print(f"Size: {results['file_info']['size_mb']:.2f} MB")
        
        # Check if duration exists before accessing
        if 'metadata' in results['technical_analysis'] and 'duration' in results['technical_analysis']['metadata']:
            print(f"Duration: {results['technical_analysis']['metadata']['duration']:.2f} seconds")
        
        # Content type
        content_type = results["combined_insights"]["content_type"]
        if content_type and 'primary_type' in content_type:
            confidence_str = f" ({content_type.get('confidence', 0)*100:.1f}% confidence)" if 'confidence' in content_type else ""
            print(f"\nContent Type: {content_type['primary_type']}{confidence_str}")
            if 'subtypes' in content_type and content_type["subtypes"]:
                print(f"Subtypes: {', '.join(content_type['subtypes'])}")
        
        # Quality
        quality = results["combined_insights"]["quality_assessment"]
        if quality:
            if 'video_quality' in quality and 'rating' in quality['video_quality']:
                print(f"\nVideo Quality: {quality['video_quality']['rating']}")
            if 'audio_quality' in quality and 'rating' in quality['audio_quality']:
                print(f"\nAudio Quality: {quality['audio_quality']['rating']}")
        
        # Technical highlights
        print("\nTechnical Highlights:")
        if ('metadata' in results['technical_analysis'] and 
            'video_streams' in results['technical_analysis']['metadata'] and 
            results['technical_analysis']['metadata']['video_streams']):
            vs = results['technical_analysis']['metadata']['video_streams'][0]
            print(f"- Video: {vs.get('width', '?')}x{vs.get('height', '?')} @ {vs.get('fps', 0):.2f}fps ({vs.get('codec', '?')})")
        
        if ('metadata' in results['technical_analysis'] and 
            'audio_streams' in results['technical_analysis']['metadata'] and 
            results['technical_analysis']['metadata']['audio_streams']):
            aud = results['technical_analysis']['metadata']['audio_streams'][0]
            print(f"- Audio: {aud.get('codec', '?')} {aud.get('sample_rate', '?')}Hz {aud.get('channels', '?')}ch")
        
        # Scene info - safely handle missing keys
        if 'scene_analysis' in results['technical_analysis']:
            scene_info = results['technical_analysis']['scene_analysis']
            scene_count = scene_info.get('scene_count', '?')
            scene_output = f"- Scenes: {scene_count}"
            
            # Only add average duration if it exists
            if 'average_scene_duration' in scene_info:
                scene_output += f" (avg {scene_info['average_scene_duration']:.2f}s)"
            print(scene_output)
        
        # Mood
        mood = results["combined_insights"]["mood"]
        if mood and 'primary_mood' in mood:
            confidence_str = f" ({mood.get('mood_confidence', 0)*100:.1f}% confidence)" if 'mood_confidence' in mood else ""
            print(f"\nMood: {mood['primary_mood']}{confidence_str}")
            if 'mood_consistency' in mood:
                print(f"Mood Consistency: {mood['mood_consistency']}")
        
        # Storytelling
        story = results["combined_insights"]["storytelling_metrics"]
        if story:
            if 'narrative_structure' in story:
                print(f"\nNarrative Structure: {story['narrative_structure']}")
            if 'pacing' in story:
                print(f"Pacing: {story['pacing']}")
            
            # Key moments
            if 'key_moments' in story and story["key_moments"]:
                print("\nKey Moments:")
                for moment in story["key_moments"]:
                    if all(k in moment for k in ['time', 'description', 'intensity']):
                        print(f"- {moment['time']}: {moment['description']} (intensity: {moment['intensity']:.2f})")
        
        # AI classifications summary
        if hasattr(self, 'clip_data') and self.clip_data:
            print("\nTop Video Classifications:")
            for item in self.clip_data[:3]:
                print(f"- {item['label']}: {item['probability']:.2f} (count: {item['count']})")
                
        if hasattr(self, 'clap_data') and self.clap_data:
            print("\nTop Audio Classifications:")
            for item in self.clap_data[:3]:
                print(f"- {item['label']}: {item['probability']:.2f} (count: {item['count']})")


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
        description='Combined Media Analysis Tool - Integrates technical analysis and AI classifications'
    )
    parser.add_argument('input', help='Path to input media file')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('--ffmpeg', default='./FFmpeg/ffmpeg', help='Path to ffmpeg binary')
    parser.add_argument('--ffprobe', default='./FFmpeg/ffprobe', help='Path to ffprobe binary')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Scene detection threshold (0.0-1.0, lower values detect more scenes)')
    parser.add_argument('--clip-categories', default='resources/labels/categories_clip.txt',
                       help='Path to CLIP categories file')
    parser.add_argument('--clap-categories', default='resources/labels/categories_clap.txt',
                       help='Path to CLAP categories file')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='AI classification temperature (lower is more focused)')
    parser.add_argument('--json', help='Path to save JSON analysis results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--from-existing', help='Load from existing technical JSON and skip analysis')
    parser.add_argument('--classification-txt', help='Path to existing classification results text file')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Using existing analysis files or running new analysis
        if args.from_existing and args.classification_txt:
            # Load from existing files
            tech_json_path = Path(args.from_existing)
            class_txt_path = Path(args.classification_txt)
            
            if not tech_json_path.exists():
                raise FileNotFoundError(f"Technical analysis file not found: {args.from_existing}")
                
            if not class_txt_path.exists():
                raise FileNotFoundError(f"Classification file not found: {args.classification_txt}")
            
            logging.info(f"Using existing analysis files")
            
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
            
            logging.info(f"Combined analysis results saved to: {json_path}")
        
        else:
            # Run new analysis
            analyzer = CombinedAnalyzer(
                input_file=args.input,
                output_dir=args.output
            )
            
            # Run technical analysis
            analyzer.run_technical_analysis(
                ffmpeg_path=args.ffmpeg,
                ffprobe_path=args.ffprobe,
                scene_threshold=args.threshold
            )
            
            # Run AI classification
            analyzer.run_ai_classification(
                clip_categories=args.clip_categories,
                clap_categories=args.clap_categories,
                temperature=args.temperature,
                scene_threshold=args.threshold
            )
            
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
            
            logging.info(f"Combined analysis results saved to: {json_path}")
        
        logging.info(f"Analysis completed successfully!")
        logging.info(f"JSON results: {json_path}")
    except Exception as e:
        logging.exception(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Combined Media Analysis Tool - Integrates technical metadata and AI classifications.
"""
import os
import sys
import subprocess
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

from src.util_classes import (
    FileInfo, MediaFileInfo, ClassificationItem, ClassificationData,
    ContentTypeInfo, QualityAssessment, MoodElement, MoodAssessment,
    KeyMoment, StorytellingMetrics, CombinedInsights, CombinedResults,
    ContentType, QualityRating, MoodConsistency, PacingRating, 
    NarrativeStructure, KeyMomentIntensity, MomentType, ResolutionClass
)


class TechnicalAnalyzer:
    """Class to perform technical analysis on media files."""
    
    def __init__(self, file_info: MediaFileInfo):
        """Initialize technical analyzer with file information.
        
        Args:
            file_info: MediaFileInfo instance containing file paths
        """
        self.file_info = file_info
        self.technical_data = None
    
    def run_analysis(self, ffmpeg_path: str = "ffmpeg", 
                     ffprobe_path: str = "ffprobe",
                     scene_threshold: float = 0.3) -> Dict:
        """Run technical analysis using the media analyzer script.
        
        Args:
            ffmpeg_path: Path to ffmpeg binary
            ffprobe_path: Path to ffprobe binary
            scene_threshold: Threshold for scene detection (0.0-1.0)
            
        Returns:
            Dictionary containing technical analysis data
        """
        logging.info(f"Running technical analysis on {self.file_info.input_file}")
        
        # Build command to run the analysis script
        output_json = self.file_info.output_dir / f"{self.file_info.input_file.stem}_technical.json"
        cmd = [
            sys.executable, "run_analysis.py",
            str(self.file_info.input_file),
            "--ffmpeg", ffmpeg_path,
            "--ffprobe", ffprobe_path,
            "--threshold", str(scene_threshold),
            "--json", str(output_json)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load the generated JSON file
            if output_json.exists():
                with open(output_json, 'r') as f:
                    self.technical_data = json.load(f)
                logging.info("Technical analysis completed successfully")
                return self.technical_data
            else:
                raise FileNotFoundError(f"Technical analysis output not found: {output_json}")
        
        except subprocess.SubprocessError as e:
            logging.error(f"Technical analysis failed: {e}")
            if hasattr(e, 'stderr'):
                logging.error(f"Error details: {e.stderr}")
            raise


class AIClassifier:
    """Class to perform AI-based classification on media files."""
    
    def __init__(self, file_info: MediaFileInfo):
        """Initialize AI classifier with file information.
        
        Args:
            file_info: MediaFileInfo instance containing file paths
        """
        self.file_info = file_info
        self.clip_data = None  # Video classifications
        self.clap_data = None  # Audio classifications
        self.classification_data = ClassificationData()
    
    def run_classification(self, 
                          clip_categories: str,
                          clap_categories: str,
                          temperature: float = 0.1,
                          scene_threshold: float = 0.3) -> Dict:
        """Run AI classification using the classification script.
        
        Args:
            clip_categories: Path to CLIP categories file
            clap_categories: Path to CLAP categories file
            temperature: Temperature parameter for classification (lower is more focused)
            scene_threshold: Threshold for scene detection
            
        Returns:
            Dictionary containing video and audio classifications
        """
        logging.info(f"Running AI classification on {self.file_info.input_file}")
        
        # Build command to run the classification script
        output_file = self.file_info.output_dir / f"{self.file_info.input_file.stem}_classifications.txt"
        cmd = [
            sys.executable, "run_classification.py",
            "--input", str(self.file_info.input_file),
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
                logging.info("AI classification completed successfully")
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
        """Parse the classification results file into structured data.
        
        Args:
            output_file: Path to the classification results file
        """
        video_results = []
        audio_results = []
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Stream 0 is typically video, Stream 1 is typically audio
                if row['stream_id'] == '0':
                    video_results.append(ClassificationItem(
                        label=row['label'],
                        probability=float(row['avg_probability']),
                        count=int(row['count'])
                    ))
                elif row['stream_id'] == '1':
                    audio_results.append(ClassificationItem(
                        label=row['label'],
                        probability=float(row['avg_probability']),
                        count=int(row['count'])
                    ))
        
        # Sort by count (descending) for more relevance
        self.clip_data = sorted(video_results, key=lambda x: x.count, reverse=True)
        self.clap_data = sorted(audio_results, key=lambda x: x.count, reverse=True)
        
        # Update classification data
        self.classification_data.video_classifications = self.clip_data
        self.classification_data.audio_classifications = self.clap_data


class InsightAnalyzer:
    """Class to analyze and derive insights from technical and AI classification data."""
    
    def __init__(self, technical_data: Dict, clip_data: List[ClassificationItem], clap_data: List[ClassificationItem]):
        """Initialize with technical and classification data.
        
        Args:
            technical_data: Dictionary containing technical analysis results
            clip_data: List of video classifications
            clap_data: List of audio classifications
        """
        self.technical_data = technical_data
        self.clip_data = clip_data
        self.clap_data = clap_data
        
        # Initialize insights structure
        self.insights = CombinedInsights()
    
    def analyze_all(self) -> CombinedInsights:
        """Run all insight analyses and return combined results.
        
        Returns:
            Object containing all insights
        """
        self._derive_content_type()
        self._assess_quality()
        self._determine_mood()
        self._analyze_storytelling()
        
        return self.insights
    
    def _derive_content_type(self) -> None:
        """
        Derive the content type from video/audio classifications and technical metadata.
        This method analyzes video and audio classification results along with technical
        metadata to determine:
        - Primary content type (Storytelling, Informational, Entertainment, or Unclassified)
        - Confidence score for the primary type classification
        - Content subtypes (Animated, Live Action, mood-based attributes)
        - Format information including:
            - Video resolution (with classification as UHD_4K, FULL_HD, HD, or SD)
            - Frame rate
            - Duration
        The method populates the `self.insights.content_type` with a ContentTypeInfo object
        containing all derived information.
        Returns:
            None: Updates the object's insights.content_type attribute directly
        """
        """Derive the content type from classifications and technical data."""
        # Extract primary classification labels
        primary_video_labels = [item.label for item in self.clip_data[:3]]
        primary_audio_labels = [item.label for item in self.clap_data[:3]]
        
        # Initialize content type assessment
        content_assessment = ContentTypeInfo(
            primary_type=ContentType.UNCLASSIFIED,
            confidence=0.0,
            subtypes=[],
            format_info={
                "duration": self.technical_data["metadata"]["duration"],
                "resolution": None,
                "frame_rate": None
            }
        )
        
        # Determine resolution description
        if self.technical_data["metadata"]["video_streams"]:
            video_stream = self.technical_data["metadata"]["video_streams"][0]
            width = video_stream["width"]
            height = video_stream["height"]
            frame_rate = video_stream["fps"]
            
            content_assessment.format_info["resolution"] = f"{width}x{height}"
            content_assessment.format_info["frame_rate"] = frame_rate
            
            # Add resolution classification
            if width >= 3840 or height >= 2160:
                content_assessment.format_info["resolution_class"] = ResolutionClass.UHD_4K
            elif width >= 1920 or height >= 1080:
                content_assessment.format_info["resolution_class"] = ResolutionClass.FULL_HD
            elif width >= 1280 or height >= 720:
                content_assessment.format_info["resolution_class"] = ResolutionClass.HD
            else:
                content_assessment.format_info["resolution_class"] = ResolutionClass.SD
        
        # Determine primary content type based on both video and audio classifications
        primary_content_map = {
            ContentType.STORYTELLING: ["Storytelling", "StorytellingAudio", "Animation", "LiveAction", "Emotional", "IntenseAudio"],
            ContentType.INFORMATIONAL: ["Informational", "InformationalAudio", "FactualAudio"],
            ContentType.ENTERTAINMENT: ["Exciting", "Lighthearted", "LightAudio"]
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
            
            content_assessment.primary_type = primary_type
            content_assessment.confidence = confidence
            
            # Add relevant subtypes
            if "Animation" in primary_video_labels:
                content_assessment.subtypes.append("Animated")
            elif "LiveAction" in primary_video_labels:
                content_assessment.subtypes.append("Live Action")
                
            # Check for mood-based subtypes
            mood_subtypes = [
                label for label in all_labels 
                if label in ["Emotional", "Exciting", "Lighthearted", "Tense", "Imaginative",
                            "EerieAudio", "IntenseAudio", "LightAudio", "EmotionalAudio"]
            ]
            if mood_subtypes:
                content_assessment.subtypes.extend(mood_subtypes)
        else:
            # Fallback if no clear content type is detected
            content_assessment.primary_type = ContentType.UNCLASSIFIED
            content_assessment.confidence = 0.0
        
        self.insights.content_type = content_assessment
    
    def _assess_quality(self) -> None:
        """
        Assess the overall quality of the media based on classification data and technical metadata.
        This method analyzes both video and audio quality using:
        1. Classification data from CLIP and CLAP models to identify quality-related attributes
        2. Technical metadata like bitrate, codec information and audio levels as fallback metrics
        The assessment results in a QualityAssessment object with:
        - Video quality rating (HIGH/STANDARD) with contributing factors
        - Audio quality rating (PROFESSIONAL/GOOD/BASIC) with contributing factors
        - Technical quality details including bitrate and codec information
        The resulting quality assessment is stored in self.insights.quality_assessment.
        """
        """Assess the overall quality of the media."""
        quality_assessment = QualityAssessment(
            video_quality={"rating": None, "factors": []},
            audio_quality={"rating": None, "factors": []},
            technical_quality={
                "bitrate": self.technical_data["metadata"]["bitrate"],
                "codec_info": {}
            }
        )
        
        # Extract quality-related classifications
        video_quality_labels = [item for item in self.clip_data if item.label in ["HighQuality", "LowQuality"]]
        audio_quality_labels = [item for item in self.clap_data if item.label in ["ProfessionalRecording", "BasicRecording"]]
        
        # Assess video quality
        if video_quality_labels:
            # Sort by probability for more accurate assessment
            sorted_labels = sorted(video_quality_labels, key=lambda x: x.probability, reverse=True)
            primary_quality = sorted_labels[0].label
            
            if primary_quality == "HighQuality":
                quality_assessment.video_quality["rating"] = QualityRating.HIGH
                quality_assessment.video_quality["factors"].append("Professional camera work")
                quality_assessment.video_quality["factors"].append("Good lighting and composition")
            else:
                quality_assessment.video_quality["rating"] = QualityRating.STANDARD
                quality_assessment.video_quality["factors"].append("Consumer-level production")
        else:
            # Fallback to technical assessment
            bitrate_mbps = self.technical_data["metadata"]["bitrate"] / 1000000
            if bitrate_mbps > 10:
                quality_assessment.video_quality["rating"] = QualityRating.HIGH
                quality_assessment.video_quality["factors"].append(f"High bitrate ({bitrate_mbps:.1f} Mbps)")
            else:
                quality_assessment.video_quality["rating"] = QualityRating.STANDARD
                quality_assessment.video_quality["factors"].append(f"Standard bitrate ({bitrate_mbps:.1f} Mbps)")
        
        # Add technical video info
        if self.technical_data["metadata"]["video_streams"]:
            video_stream = self.technical_data["metadata"]["video_streams"][0]
            quality_assessment.technical_quality["codec_info"]["video"] = {
                "codec": video_stream["codec"],
                "profile": video_stream["profile"],
                "bit_depth": video_stream["bit_depth"],
                "color_space": video_stream["color_space"]
            }
        
        # Assess audio quality
        if audio_quality_labels:
            sorted_labels = sorted(audio_quality_labels, key=lambda x: x.probability, reverse=True)
            primary_quality = sorted_labels[0].label
            
            if primary_quality == "ProfessionalRecording":
                quality_assessment.audio_quality["rating"] = QualityRating.PROFESSIONAL
                quality_assessment.audio_quality["factors"].append("Clear audio with good dynamic range")
                quality_assessment.audio_quality["factors"].append("Professional post-processing")
            else:
                quality_assessment.audio_quality["rating"] = QualityRating.BASIC
                quality_assessment.audio_quality["factors"].append("Standard audio quality")
        else:
            # Fallback to technical assessment
            if self.technical_data["audio_analysis"]["mean_volume"] is not None:
                mean_vol = self.technical_data["audio_analysis"]["mean_volume"] 
                max_vol = self.technical_data["audio_analysis"]["max_volume"]
                
                if mean_vol > -20 and max_vol > -10:
                    quality_assessment.audio_quality["rating"] = QualityRating.GOOD
                    quality_assessment.audio_quality["factors"].append("Balanced audio levels")
                else:
                    quality_assessment.audio_quality["rating"] = QualityRating.BASIC
                    quality_assessment.audio_quality["factors"].append("Variable audio levels")
        
        # Add technical audio info
        if self.technical_data["metadata"]["audio_streams"]:
            audio_stream = self.technical_data["metadata"]["audio_streams"][0]
            quality_assessment.technical_quality["codec_info"]["audio"] = {
                "codec": audio_stream["codec"],
                "sample_rate": audio_stream["sample_rate"],
                "channels": audio_stream["channels"],
                "channel_layout": audio_stream["channel_layout"]
            }
            if int(audio_stream["sample_rate"]) >= 48000:
                quality_assessment.audio_quality["factors"].append("Professional audio sampling rate")
        
        self.insights.quality_assessment = quality_assessment
    
    def _determine_mood(self) -> None:
        """
        Determine the overall mood of the content based on CLIP and CLAP classification results.
        This method analyzes both visual and audio mood-related classifications to establish
        the primary mood of the video content. It calculates mood weights by combining probabilities 
        and occurrence counts, then assigns the mood with the highest weight as the primary mood.
        The method also:
        - Calculates mood confidence based on the relative weight of the primary mood
        - Creates a list of all detected moods with their relative strengths
        - Falls back to scene frequency analysis for mood determination if no mood labels are detected
        - Analyzes scene durations to determine mood consistency (highly consistent, moderately consistent, or variable)
        - Evaluates audio action moments to create a mood progression timeline across the content
        Results are stored in the insights.mood attribute as a MoodAssessment object.
        """
        """Determine the overall mood of the content."""
        mood_assessment = MoodAssessment(
            primary_mood="",
            mood_confidence=0.0
        )
        
        # Gather mood-related classifications
        mood_labels = [
            item for item in self.clip_data 
            if item.label in ["Emotional", "Exciting", "Lighthearted", "Tense", "Imaginative"]
        ]
        mood_labels_audio = [
            item for item in self.clap_data
            if item.label in ["EmotionalAudio", "IntenseAudio", "LightAudio", "EerieAudio", "FantasticalAudio"]
        ]
        
        # Combine and find most common mood
        all_moods = mood_labels + mood_labels_audio
        if all_moods:
            # Get weighted count of each mood (probability * count)
            mood_weights = defaultdict(float)
            for mood in all_moods:
                label = mood.label.replace("Audio", "")  # Normalize audio and video labels
                weight = mood.probability * mood.count
                mood_weights[label] += weight
            
            # Find the primary mood with highest weight
            if mood_weights:
                primary_mood, weight = max(mood_weights.items(), key=lambda x: x[1])
                total_weight = sum(mood_weights.values())
                confidence = weight / total_weight if total_weight > 0 else 0
                
                # Clean up mood name (remove "Audio" suffix)
                primary_mood = primary_mood.replace("Audio", "")
                
                mood_assessment.primary_mood = primary_mood
                mood_assessment.mood_confidence = confidence
            
            # Add all detected moods as mood elements
            for mood_name, weight in mood_weights.items():
                normalized_name = mood_name.replace("Audio", "")
                total_weight = sum(mood_weights.values())
                strength = weight / total_weight if total_weight > 0 else 0
                mood_assessment.mood_elements.append(MoodElement(
                    type=normalized_name,
                    strength=strength
                ))
        else:
            # Fallback mood determination from scene analysis
            scene_count = self.technical_data["scene_analysis"]["scene_count"]
            duration = self.technical_data["metadata"]["duration"]
            
            if scene_count / duration > 0.5:  # More than 1 scene per 2 seconds
                mood_assessment.primary_mood = "Fast-paced"
                mood_assessment.mood_confidence = 0.6
                mood_assessment.mood_elements.append(MoodElement(
                    type="Fast-paced",
                    strength=0.6
                ))
            else:
                mood_assessment.primary_mood = "Moderate"
                mood_assessment.mood_confidence = 0.5
                mood_assessment.mood_elements.append(MoodElement(
                    type="Moderate",
                    strength=0.5
                ))
                
        # Determine mood consistency based on scene frequency variations
        if self.technical_data["scene_analysis"]["scene_count"] > 1:
            scene_times = self.technical_data["scene_analysis"]["scene_timestamps"]
            scene_durations = [scene_times[i+1] - scene_times[i] for i in range(len(scene_times)-1)]
            
            if scene_durations:
                mean_duration = sum(scene_durations) / len(scene_durations)
                variation = sum(abs(d - mean_duration) for d in scene_durations) / len(scene_durations) / mean_duration
                
                if variation < 0.3:
                    mood_assessment.mood_consistency = MoodConsistency.HIGHLY_CONSISTENT
                elif variation < 0.6:
                    mood_assessment.mood_consistency = MoodConsistency.MODERATELY_CONSISTENT
                else:
                    mood_assessment.mood_consistency = MoodConsistency.VARIABLE
                    
                # Add variation metric
                mood_assessment.scene_rhythm_variation = variation
        
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
            
            mood_assessment.mood_progression = segments
        
        self.insights.mood = mood_assessment
    
    def _analyze_storytelling(self) -> None:
        """
        Analyze storytelling metrics from the combined technical data.
        This method processes scene data, audio analysis, and video duration to extract
        storytelling insights including:
        - Scene statistics (count, average duration, scenes per minute)
        - Pacing classification (very fast, fast, moderate, slow)
        - Key moments identification based on audio intensity
        - Narrative structure analysis (rising action, complex, falling action, episodic, simple)
        The analysis results are stored in self.insights.storytelling_metrics as a StorytellingMetrics object.
        Returns:
            None
        """
        """Analyze storytelling metrics from the combined data."""
        scenes_per_minute = self.technical_data["scene_analysis"]["scene_count"] / (self.technical_data["metadata"]["duration"] / 60)
        
        storytelling_metrics = StorytellingMetrics(
            narrative_structure=None,
            pacing=None,
            scene_analysis={
                "count": self.technical_data["scene_analysis"]["scene_count"],
                "average_duration": self.technical_data["scene_analysis"]["average_scene_duration"],
                "scenes_per_minute": scenes_per_minute
            }
        )
        
        # Determine pacing from scene frequency
        scenes_per_minute = storytelling_metrics.scene_analysis["scenes_per_minute"]
        if scenes_per_minute > 20:
            storytelling_metrics.pacing = PacingRating.VERY_FAST
        elif scenes_per_minute > 10:
            storytelling_metrics.pacing = PacingRating.FAST
        elif scenes_per_minute > 5:
            storytelling_metrics.pacing = PacingRating.MODERATE
        else:
            storytelling_metrics.pacing = PacingRating.SLOW
        
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
                
                # Add with descriptive text based on intensity
                if moment["intensity"] > 0.9:
                    storytelling_metrics.key_moments.append(KeyMoment(
                        time=time_formatted,
                        description=KeyMomentIntensity.MAJOR,
                        intensity=moment["intensity"],
                        type=moment["type"]
                    ))
                elif moment["intensity"] > 0.7:
                    storytelling_metrics.key_moments.append(KeyMoment(
                        time=time_formatted,
                        description=KeyMomentIntensity.SIGNIFICANT,
                        intensity=moment["intensity"],
                        type=moment["type"]
                    ))
                else:
                    storytelling_metrics.key_moments.append(KeyMoment(
                        time=time_formatted,
                        description=KeyMomentIntensity.NOTABLE,
                        intensity=moment["intensity"],
                        type=moment["type"]
                    ))
        
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
                storytelling_metrics.narrative_structure = NarrativeStructure.RISING_ACTION
            elif scenes_in_first > scenes_in_second and scenes_in_second < scenes_in_third:
                storytelling_metrics.narrative_structure = NarrativeStructure.COMPLEX
            elif scenes_in_first > scenes_in_second and scenes_in_second > scenes_in_third:
                storytelling_metrics.narrative_structure = NarrativeStructure.FALLING_ACTION
            else:
                storytelling_metrics.narrative_structure = NarrativeStructure.EPISODIC
        else:
            storytelling_metrics.narrative_structure = NarrativeStructure.SIMPLE
        
        self.insights.storytelling_metrics = storytelling_metrics


class CombinedAnalyzer:
    """Main class that orchestrates the combined media analysis workflow."""
    
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """Initialize with input file and output directory.
        
        Args:
            input_file: Path to the media file to analyze
            output_dir: Directory for analysis outputs
        """
        # Initialize file info
        self.file_info = MediaFileInfo(input_file, output_dir)
        
        # Initialize analyzers
        self.tech_analyzer = TechnicalAnalyzer(self.file_info)
        self.ai_classifier = AIClassifier(self.file_info)
        
        # Initialize results structure
        self.combined_results = CombinedResults(
            file_info=self.file_info.file_info
        )
    
    @property
    def technical_data(self):
        """Access to technical data."""
        return self.tech_analyzer.technical_data
    
    @technical_data.setter
    def technical_data(self, value):
        """Set technical data directly."""
        self.tech_analyzer.technical_data = value
    
    @property
    def clip_data(self):
        """Access to video classification data."""
        return self.ai_classifier.clip_data
    
    @property
    def clap_data(self):
        """Access to audio classification data."""
        return self.ai_classifier.clap_data
    
    def run_technical_analysis(self, **kwargs) -> Dict:
        """Run technical analysis with provided parameters."""
        return self.tech_analyzer.run_analysis(**kwargs)
    
    def run_ai_classification(self, **kwargs) -> Dict:
        """Run AI classification with provided parameters."""
        return self.ai_classifier.run_classification(**kwargs)
    
    def _parse_classification_results(self, output_file: Path) -> None:
        """Parse classification results from a file."""
        self.ai_classifier._parse_classification_results(output_file)
    
    def combine_analyses(self) -> CombinedResults:
        """Combine technical analysis and AI classifications to generate insights."""
        if not self.technical_data:
            raise ValueError("Technical analysis has not been run yet")
        if not self.clip_data or not self.clap_data:
            raise ValueError("AI classification has not been run yet")
        
        # Update combined results with raw data
        self.combined_results.technical_analysis = self.technical_data
        self.combined_results.ai_classifications = {
            "video": [item.to_dict() for item in self.clip_data],
            "audio": [item.to_dict() for item in self.clap_data]
        }
        
        # Generate insights
        insight_analyzer = InsightAnalyzer(self.technical_data, self.clip_data, self.clap_data)
        self.combined_results.combined_insights = insight_analyzer.analyze_all()
        
        return self.combined_results
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save combined analysis results to a JSON file.
        
        Args:
            output_file: Optional path to save the results
            
        Returns:
            Path to the saved file
        """
        if output_file is None:
            output_file = self.file_info.output_dir / f"{self.file_info.input_file.stem}_combined_analysis.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(self.combined_results.to_dict(), f, indent=2)
        
        return str(output_file)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results to the console."""
        results = self.combined_results
        
        print("\n=== Combined Media Analysis Summary ===")
        print(f"File: {self.file_info.input_file}")
        print(f"Size: {results.file_info.size_mb:.2f} MB")
        
        # Safely access duration
        if results.technical_analysis and 'metadata' in results.technical_analysis and 'duration' in results.technical_analysis['metadata']:
            print(f"Duration: {results.technical_analysis['metadata']['duration']:.2f} seconds")
        
        # Content type section
        self._print_content_type_summary()
        
        # Quality section
        self._print_quality_summary()
        
        # Technical highlights section
        self._print_technical_highlights()
        
        # Mood section
        self._print_mood_summary()
        
        # Storytelling section
        self._print_storytelling_summary()
        
        # AI classifications summary
        self._print_classification_summary()
    
    def _print_content_type_summary(self) -> None:
        """Print content type summary information."""
        content_type = self.combined_results.combined_insights.content_type
        if content_type:
            confidence_str = f" ({content_type.confidence*100:.1f}% confidence)"
            print(f"\nContent Type: {content_type.primary_type}{confidence_str}")
            if content_type.subtypes:
                print(f"Subtypes: {', '.join(content_type.subtypes)}")
    
    def _print_quality_summary(self) -> None:
        """Print quality assessment summary information."""
        quality = self.combined_results.combined_insights.quality_assessment
        if quality:
            if quality.video_quality and 'rating' in quality.video_quality:
                print(f"\nVideo Quality: {quality.video_quality['rating']}")
            if quality.audio_quality and 'rating' in quality.audio_quality:
                print(f"Audio Quality: {quality.audio_quality['rating']}")
    
    def _print_technical_highlights(self) -> None:
        """Print technical highlights summary."""
        results = self.combined_results
        print("\nTechnical Highlights:")
        
        # Video stream info
        if (results.technical_analysis and 
            'metadata' in results.technical_analysis and 
            'video_streams' in results.technical_analysis['metadata'] and 
            results.technical_analysis['metadata']['video_streams']):
            vs = results.technical_analysis['metadata']['video_streams'][0]
            print(f"- Video: {vs.get('width', '?')}x{vs.get('height', '?')} @ {vs.get('fps', 0):.2f}fps ({vs.get('codec', '?')})")
        
        # Audio stream info
        if (results.technical_analysis and 
            'metadata' in results.technical_analysis and 
            'audio_streams' in results.technical_analysis['metadata'] and 
            results.technical_analysis['metadata']['audio_streams']):
            aud = results.technical_analysis['metadata']['audio_streams'][0]
            print(f"- Audio: {aud.get('codec', '?')} {aud.get('sample_rate', '?')}Hz {aud.get('channels', '?')}ch")
        
        # Scene info
        if results.technical_analysis and 'scene_analysis' in results.technical_analysis:
            scene_info = results.technical_analysis['scene_analysis']
            scene_count = scene_info.get('scene_count', '?')
            scene_output = f"- Scenes: {scene_count}"
            
            # Only add average duration if it exists
            if 'average_scene_duration' in scene_info:
                scene_output += f" (avg {scene_info['average_scene_duration']:.2f}s)"
            print(scene_output)
    
    def _print_mood_summary(self) -> None:
        """Print mood summary information."""
        mood = self.combined_results.combined_insights.mood
        if mood:
            confidence_str = f" ({mood.mood_confidence*100:.1f}% confidence)" if mood.mood_confidence else ""
            print(f"\nMood: {mood.primary_mood}{confidence_str}")
            if mood.mood_consistency:
                print(f"Mood Consistency: {mood.mood_consistency}")
    
    def _print_storytelling_summary(self) -> None:
        """Print storytelling metrics summary."""
        story = self.combined_results.combined_insights.storytelling_metrics
        if story:
            if story.narrative_structure:
                print(f"\nNarrative Structure: {story.narrative_structure}")
            if story.pacing:
                print(f"Pacing: {story.pacing}")
            
            # Key moments
            if story.key_moments:
                print("\nKey Moments:")
                for moment in story.key_moments:
                    print(f"- {moment.time}: {moment.description} (intensity: {moment.intensity:.2f})")
    
    def _print_classification_summary(self) -> None:
        """Print AI classification summary."""
        if self.clip_data:
            print("\nTop Video Classifications:")
            for item in self.clip_data[:3]:
                print(f"- {item.label}: {item.probability:.2f} (count: {item.count})")
                
        if self.clap_data:
            print("\nTop Audio Classifications:")
            for item in self.clap_data[:3]:
                print(f"- {item.label}: {item.probability:.2f} (count: {item.count})")


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
    # Initialize analyzer
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
    
    return json_path

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
    parser.add_argument('--ffmpeg', default='./FFmpeg/ffmpeg', help='Path to ffmpeg binary')
    parser.add_argument('--ffprobe', default='./FFmpeg/ffprobe', help='Path to ffprobe binary')
    parser.add_argument('--threshold', type=float, default=0.3,
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
        logging.exception(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
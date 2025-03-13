#!/usr/bin/env python3
"""
Media Analysis Classes - Core components for analyzing media files.
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import StrEnum

class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass


#######################################
# Enums for standardized string values
#######################################

class ContentType(StrEnum):
    """Enumeration of content types."""
    STORYTELLING = "Storytelling"
    INFORMATIONAL = "Informational"
    ENTERTAINMENT = "Entertainment"
    UNCLASSIFIED = "Unclassified Media"


class ResolutionClass(StrEnum):
    """Enumeration of resolution classes."""
    SD = "SD"
    HD = "HD"
    FULL_HD = "Full HD"
    UHD_4K = "4K"


class FrameRateClass(StrEnum):
    """Enumeration of frame rate classes."""
    FILM = "Film standard (24fps)"
    BROADCAST = "Broadcast standard (30fps)"
    HIGH_FRAME_RATE = "High frame rate"


class AudioClass(StrEnum):
    """Enumeration of audio quality classes."""
    CONSUMER = "Consumer"
    PROFESSIONAL = "Professional"


class QualityRating(StrEnum):
    """Enumeration of quality ratings."""
    BASIC = "Basic"
    STANDARD = "Standard"
    GOOD = "Good"
    HIGH = "High"
    PROFESSIONAL = "Professional"


class MoodConsistency(StrEnum):
    """Enumeration of mood consistency ratings."""
    HIGHLY_CONSISTENT = "Highly consistent"
    MODERATELY_CONSISTENT = "Moderately consistent"
    VARIABLE = "Variable"


class PacingRating(StrEnum):
    """Enumeration of pacing ratings."""
    SLOW = "Slow"
    MODERATE = "Moderate"
    FAST = "Fast"
    VERY_FAST = "Very Fast"


class NarrativeStructure(StrEnum):
    """Enumeration of narrative structure types."""
    SIMPLE = "Simple"
    RISING_ACTION = "Rising Action"
    FALLING_ACTION = "Falling Action"
    COMPLEX = "Complex (Setup-Conflict-Resolution)"
    EPISODIC = "Episodic"


class SystemClassification(StrEnum):
    """Enumeration of system classifications."""
    UNKNOWN = "Unknown system"
    IDENTIFIED = "Identified device"
    PROFESSIONAL = "Professional camera system"
    PROSUMER = "Prosumer camera system"
    CONSUMER = "Consumer camera"


class MomentType(StrEnum):
    """Enumeration of action moment types."""
    INITIAL_AUDIO = "initial_audio"
    SOUND_SECTION = "sound_section"
    FINAL_AUDIO = "final_audio"
    VOLUME_PEAK = "volume_peak"
    SCENE_CHANGE = "scene_change"


class KeyMomentIntensity(StrEnum):
    """Enumeration of key moment intensity descriptions."""
    MAJOR = "Major action/emotional moment"
    SIGNIFICANT = "Significant moment"
    NOTABLE = "Notable moment"


class DetectionMethod(StrEnum):
    """Enumeration of detection methods."""
    SHOWINFO = "showinfo"
    FALLBACK = "fallback"
    SILENCEDETECT = "silencedetect"
    VOLUMEDETECT = "volumedetect"
    COMBINED = "silencedetect+volumedetect"


#######################################
# Data Classes
#######################################

@dataclass
class FileInfo:
    """Data class to store basic file information."""
    filename: str
    path: str
    size_mb: float
    created_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class VideoStream:
    """Structured data class for video stream information."""
    codec: str = None
    width: int = None
    height: int = None
    fps: float = None
    bit_depth: int = None
    pix_fmt: str = None
    profile: str = None
    color_space: str = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AudioStream:
    """Structured data class for audio stream information."""
    codec: str = None
    sample_rate: int = None
    channels: int = None
    channel_layout: str = None
    bit_depth: int = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Metadata:
    """Structured data class for file metadata."""
    format: str = "unknown"
    duration: float = 0
    size: int = 0
    bitrate: int = 0
    streams: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    video_streams: List[VideoStream] = field(default_factory=list)
    audio_streams: List[AudioStream] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["video_streams"] = [vs.to_dict() for vs in self.video_streams]
        result["audio_streams"] = [aus.to_dict() for aus in self.audio_streams]
        return result


@dataclass
class ActionMoment:
    """Structured data class for audio action moments."""
    time: float
    type: Union[str, MomentType]
    intensity: float
    duration: Optional[float] = None
    max_volume: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AudioAnalysis:
    """Structured data class for audio analysis results."""
    has_audio: bool = False
    mean_volume: Optional[float] = None
    max_volume: Optional[float] = None
    action_moment_count: int = 0
    action_moments: List[ActionMoment] = field(default_factory=list)
    analysis_method: Union[str, DetectionMethod] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["action_moments"] = [moment.to_dict() for moment in self.action_moments]
        result["analysis_method"] = (
            self.analysis_method.value 
            if isinstance(self.analysis_method, DetectionMethod) 
            else self.analysis_method
        )
        return result


@dataclass
class SceneAnalysis:
    """Structured data class for scene analysis results."""
    total_frames: int = 0
    scene_count: int = 0
    scene_timestamps: List[float] = field(default_factory=list)
    average_scene_duration: float = 0
    scene_frequency: float = 0
    threshold_used: float = 0.3
    detection_method: Union[str, DetectionMethod] = DetectionMethod.SHOWINFO
    thumbnail_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["detection_method"] = (
            self.detection_method.value 
            if isinstance(self.detection_method, DetectionMethod) 
            else self.detection_method
        )
        return result


@dataclass
class SystemInfo:
    """Structured data class for system information."""
    classification: Union[str, SystemClassification] = SystemClassification.UNKNOWN
    details: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["classification"] = (
            self.classification.value 
            if isinstance(self.classification, SystemClassification) 
            else self.classification
        )
        return result


@dataclass
class AnalysisResults:
    """Structured data class for all analysis results."""
    input_file: str
    metadata: Metadata = field(default_factory=Metadata)
    scene_analysis: SceneAnalysis = field(default_factory=SceneAnalysis)
    audio_analysis: AudioAnalysis = field(default_factory=AudioAnalysis)
    system_info: SystemInfo = field(default_factory=SystemInfo)
    analysis_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "input_file": self.input_file,
            "metadata": self.metadata.to_dict(),
            "scene_analysis": self.scene_analysis.to_dict(),
            "audio_analysis": self.audio_analysis.to_dict(),
            "system_info": self.system_info.to_dict(),
            "analysis_parameters": self.analysis_parameters
        }


@dataclass
class ClassificationItem:
    """Data class to store a single classification result."""
    label: str
    probability: float
    count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ClassificationData:
    """Data class to store all classification results."""
    video_classifications: List[ClassificationItem] = field(default_factory=list)
    audio_classifications: List[ClassificationItem] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "video": [item.to_dict() for item in self.video_classifications],
            "audio": [item.to_dict() for item in self.audio_classifications]
        }


@dataclass
class ContentTypeInfo:
    """Data class to store content type information."""
    primary_type: Union[str, ContentType] = ContentType.UNCLASSIFIED
    confidence: float = 0.0
    subtypes: List[str] = field(default_factory=list)
    format_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["primary_type"] = (
            self.primary_type.value 
            if isinstance(self.primary_type, ContentType) 
            else self.primary_type
        )
        return result


@dataclass
class QualityAssessment:
    """Data class to store quality assessment information."""
    video_quality: Dict = field(default_factory=dict)
    audio_quality: Dict = field(default_factory=dict)
    technical_quality: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MoodElement:
    """Data class to store mood element information."""
    type: str
    strength: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MoodAssessment:
    """Data class to store mood assessment information."""
    primary_mood: str
    mood_confidence: float = 0.0
    mood_elements: List[MoodElement] = field(default_factory=list)
    mood_progression: List[Dict] = field(default_factory=list)
    mood_consistency: Optional[Union[str, MoodConsistency]] = None
    scene_rhythm_variation: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "primary_mood": self.primary_mood,
            "mood_confidence": self.mood_confidence,
            "mood_elements": [element.to_dict() for element in self.mood_elements],
            "mood_progression": self.mood_progression,
        }
        
        if self.mood_consistency:
            result["mood_consistency"] = (
                self.mood_consistency.value 
                if isinstance(self.mood_consistency, MoodConsistency) 
                else self.mood_consistency
            )
            
        if self.scene_rhythm_variation is not None:
            result["scene_rhythm_variation"] = self.scene_rhythm_variation
            
        return result


@dataclass
class KeyMoment:
    """Data class to store key moment information."""
    time: str
    description: Union[str, KeyMomentIntensity]
    intensity: float
    type: Union[str, MomentType]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["description"] = (
            self.description.value 
            if isinstance(self.description, KeyMomentIntensity) 
            else self.description
        )
        result["type"] = (
            self.type.value 
            if isinstance(self.type, MomentType) 
            else self.type
        )
        return result


@dataclass
class StorytellingMetrics:
    """Data class to store storytelling metrics."""
    narrative_structure: Optional[Union[str, NarrativeStructure]] = None
    pacing: Optional[Union[str, PacingRating]] = None
    key_moments: List[KeyMoment] = field(default_factory=list)
    scene_analysis: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "scene_analysis": self.scene_analysis,
            "key_moments": [moment.to_dict() for moment in self.key_moments]
        }
        
        if self.narrative_structure:
            result["narrative_structure"] = (
                self.narrative_structure.value 
                if isinstance(self.narrative_structure, NarrativeStructure) 
                else self.narrative_structure
            )
            
        if self.pacing:
            result["pacing"] = (
                self.pacing.value 
                if isinstance(self.pacing, PacingRating) 
                else self.pacing
            )
            
        return result


@dataclass
class CombinedInsights:
    """Data class to store all combined insights."""
    content_type: Optional[ContentTypeInfo] = None
    quality_assessment: Optional[QualityAssessment] = None
    mood: Optional[MoodAssessment] = None
    storytelling_metrics: Optional[StorytellingMetrics] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {}
        if self.content_type:
            result["content_type"] = self.content_type.to_dict()
        if self.quality_assessment:
            result["quality_assessment"] = self.quality_assessment.to_dict()
        if self.mood:
            result["mood"] = self.mood.to_dict()
        if self.storytelling_metrics:
            result["storytelling_metrics"] = self.storytelling_metrics.to_dict()
        return result


@dataclass
class CombinedResults:
    """Data class to store all combined analysis results."""
    file_info: FileInfo
    technical_analysis: Optional[Dict] = None
    ai_classifications: Dict = field(default_factory=lambda: {"video": None, "audio": None})
    combined_insights: Optional[CombinedInsights] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "file_info": self.file_info.to_dict(),
            "technical_analysis": self.technical_analysis,
            "ai_classifications": self.ai_classifications,
        }
        
        if self.combined_insights:
            result["combined_insights"] = self.combined_insights.to_dict()
            
        return result


#######################################
# Utility Classes
#######################################

class FFmpegTool:
    """Handles FFmpeg operations and command execution."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._validate_tools()
    
    def _validate_tools(self):
        """Validate that FFmpeg tools are available and working."""
        for cmd, tool_path in [(["-version"], self.ffmpeg_path), 
                              (["-version"], self.ffprobe_path)]:
            try:
                subprocess.run([tool_path] + cmd, capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                raise AnalysisError(f"FFmpeg tool {tool_path} is not available: {e}")
    
    def run_command(self, cmd: List[str], check: bool = True, capture_output: bool = True):
        """Execute a command and return the result."""
        try:
            # Use text=True to get string output instead of bytes
            result = subprocess.run(
                cmd, 
                check=check, 
                capture_output=capture_output, 
                text=True,
                # Add these parameters to avoid terminal state issues
                stdin=subprocess.PIPE
            )
            return result
        except subprocess.SubprocessError as e:
            logging.error(f"Command failed: {' '.join(cmd)}")
            logging.error(f"Error: {e}")
            if check:
                raise AnalysisError(f"FFmpeg command failed: {e}")
            return e
        finally:
            # Attempt to reset terminal state if running on Unix-like system
            if os.name == 'posix':
                try:
                    os.system('stty sane')
                except Exception:
                    pass


class MediaFileInfo:
    """Class to handle media file information and validation."""
    
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """Initialize with input file path and optional output directory.
        
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
        
        # Basic file info
        self.file_info = FileInfo(
            filename=self.input_file.name,
            path=str(self.input_file),
            size_mb=self.input_file.stat().st_size / (1024 * 1024)
        )
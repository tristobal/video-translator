#!/usr/bin/env python3
"""
Video Translation and Dubbing Tool

A tool for translating and dubbing videos from English to Spanish.
Supports subtitle generation and voice dubbing using AI models.
"""

import os
import subprocess
import argparse
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

import whisper
import requests
import edge_tts


@dataclass
class TranscriptionSegment:
    """Represents a transcribed audio segment with timing information."""
    text: str
    start: float
    end: float


@dataclass
class Config:
    """Configuration settings for the video translator."""
    input_video: Path = Path("input/gardeners.world.2025.episode.01.S5801.mp4")
    output_audio_dir = Path("output/gardeners.world.2025.episode.01.S5801.wav")
    output_dir: Path = Path("output")
    groq_api_key: str = "tu_api_key"
    whisper_model: str = "medium"
    edge_voice: str = "es-ES-ElviraNeural"  # Spanish female voice
    speech_rate: str = "+0%"  # Normal speed
    speech_pitch: str = "+0Hz"  # Normal pitch
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(exist_ok=True)
        self.input_video.parent.mkdir(exist_ok=True)


class VideoTranslatorError(Exception):
    """Base exception for video translator errors."""
    pass


class AudioExtractionError(VideoTranslatorError):
    """Raised when audio extraction fails."""
    pass


class TranscriptionError(VideoTranslatorError):
    """Raised when transcription fails."""
    pass


class TranslationError(VideoTranslatorError):
    """Raised when translation fails."""
    pass


class TTSError(VideoTranslatorError):
    """Raised when text-to-speech synthesis fails."""
    pass


class EdgeTTSError(VideoTranslatorError):
    """Raised when Edge TTS synthesis fails."""
    pass


class BaseProcessor(ABC):
    """Abstract base class for processing steps."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process the input and return the result."""
        pass


class AudioExtractor(BaseProcessor):
    """Handles audio extraction from video files."""
    
    def process(self, input_video: Path, output_audio: Path) -> None:
        """Extract audio from video file."""
        try:
            self.logger.info(f"Extracting audio from {input_video}")
            subprocess.run([
                "ffmpeg", "-i", str(input_video), 
                "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", str(output_audio)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(f"Failed to extract audio: {e}")


class AudioTranscriber(BaseProcessor):
    """Handles audio transcription using Whisper."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._model = None
    
    @property
    def model(self):
        """Lazy loading of Whisper model."""
        if self._model is None:
            self.logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self._model = whisper.load_model(self.config.whisper_model)
        return self._model
    
    def process(self, audio_file: Path) -> List[TranscriptionSegment]:
        """Transcribe audio file and return segments."""
        try:
            self.logger.info(f"Transcribing {audio_file}")
            result = self.model.transcribe(str(audio_file), language="en")
            
            # Save full transcript
            transcript_file = self.config.output_dir / "transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            return [
                TranscriptionSegment(
                    text=segment["text"],
                    start=segment["start"],
                    end=segment["end"]
                )
                for segment in result["segments"]
            ]
        except Exception as e:
            raise TranscriptionError(f"Failed to transcribe audio: {e}")


class TextTranslator(BaseProcessor):
    """Handles text translation using Groq API."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }
    
    def process(self, text: str) -> str:
        """Translate text from English to Spanish."""
        if not text.strip():
            return ""
        
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "Traduce al español de forma natural y neutra."},
                {"role": "user", "content": text}
            ]
        }
        
        try:
            self.logger.debug(f"Translating: {text[:50]}...")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise TranslationError(f"Failed to translate text: {e}")
        except (KeyError, IndexError) as e:
            raise TranslationError(f"Invalid response format: {e}")


class SubtitleGenerator(BaseProcessor):
    """Generates SRT subtitle files."""
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to SRT timestamp format."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}".replace('.', ',')
    
    def process(self, segments: List[TranscriptionSegment], translations: List[str]) -> Path:
        """Generate SRT subtitle file."""
        srt_file = self.config.output_dir / "translated.srt"
        
        self.logger.info(f"Generating subtitles: {srt_file}")
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, (segment, translation) in enumerate(zip(segments, translations), 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_timestamp(segment.start)} --> "
                        f"{self._format_timestamp(segment.end)}\n")
                f.write(f"{translation.strip()}\n\n")
        
        return srt_file


class VoiceSynthesizer(BaseProcessor):
    """Handles text-to-speech synthesis using Edge TTS."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.voice = config.edge_voice
        self.rate = config.speech_rate
        self.pitch = config.speech_pitch
    
    async def _synthesize_async(self, text: str, output_file: Path) -> None:
        """Asynchronously synthesize speech from text."""
        communicate = edge_tts.Communicate(
            text, 
            self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        await communicate.save(str(output_file))
    
    def process(self, text: str) -> Path:
        """Synthesize speech from text."""
        if not text.strip():
            raise EdgeTTSError("Cannot synthesize empty text")
        
        dubbed_audio = self.config.output_dir / "dubbed.wav"
        
        try:
            self.logger.info(f"Synthesizing audio with voice '{self.voice}': {dubbed_audio}")
            # Run the async function in the current event loop or create a new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, use asyncio.create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._synthesize_async(text, dubbed_audio)
                        )
                        future.result()
                else:
                    loop.run_until_complete(self._synthesize_async(text, dubbed_audio))
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(self._synthesize_async(text, dubbed_audio))
            
            return dubbed_audio
        except Exception as e:
            raise EdgeTTSError(f"Failed to synthesize audio: {e}")
    
    @staticmethod
    async def get_available_voices() -> List[Dict]:
        """Get list of available Edge TTS voices."""
        voices = await edge_tts.list_voices()
        return voices
    
    @classmethod
    def list_spanish_voices(cls) -> None:
        """Print available Spanish voices."""
        try:
            voices = asyncio.run(cls.get_available_voices())
            spanish_voices = [
                voice for voice in voices 
                if voice['Locale'].startswith('es-')
            ]
            
            print("\nAvailable Spanish voices:")
            print("-" * 50)
            for voice in spanish_voices:
                print(f"Name: {voice['ShortName']}")
                print(f"Gender: {voice['Gender']}")
                print(f"Locale: {voice['Locale']}")
                print(f"Description: {voice.get('LocalName', 'N/A')}")
                print("-" * 30)
        except Exception as e:
            print(f"Error fetching voices: {e}")


class VideoMerger(BaseProcessor):
    """Handles video and audio merging."""
    
    def process(self, input_video: Path, dubbed_audio: Path) -> Path:
        """Merge video with dubbed audio."""
        final_video = self.config.output_dir / "final_video.mp4"
        
        try:
            self.logger.info(f"Merging video with dubbed audio: {final_video}")
            subprocess.run([
                "ffmpeg", "-i", str(input_video), "-i", str(dubbed_audio),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", 
                "-shortest", str(final_video)
            ], check=True, capture_output=True)
            return final_video
        except subprocess.CalledProcessError as e:
            raise VideoTranslatorError(f"Failed to merge video: {e}")


class VideoTranslator:
    """Main class orchestrating the video translation process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize processors
        self.audio_extractor = AudioExtractor(config)
        self.transcriber = AudioTranscriber(config)
        self.translator = TextTranslator(config)
        self.subtitle_generator = SubtitleGenerator(config)
        self.voice_synthesizer = VoiceSynthesizer(config)
        self.video_merger = VideoMerger(config)
    
    def extract_and_transcribe(self) -> List[TranscriptionSegment]:
        """Extract audio and transcribe it."""
        #audio_file = self.config.output_dir / "audio.wav"
        #self.audio_extractor.process(self.config.input_video, audio_file)
        audio_file = self.config.output_audio_dir
        return self.transcriber.process(audio_file)
    
    def translate_segments(self, segments: List[TranscriptionSegment]) -> List[str]:
        """Translate all segments."""
        self.logger.info(f"Translating {len(segments)} segments...")
        return [self.translator.process(segment.text) for segment in segments]
    
    def generate_subtitles(
            self,
            segments: List[TranscriptionSegment], 
            translations: List[str]
        ) -> Path:
        """Generate subtitle file."""
        return self.subtitle_generator.process(segments, translations)
    
    def generate_dubbing(self, translations: List[str]) -> Path:
        """Generate dubbed video."""
        full_translation = " ".join(translations)
        dubbed_audio = self.voice_synthesizer.process(full_translation)
        return self.video_merger.process(self.config.input_video, dubbed_audio)
    
    def process(self, generate_srt: bool = False, generate_dub: bool = False) -> Dict[str, Optional[Path]]:
        """Main processing method."""
        if not generate_srt and not generate_dub:
            raise ValueError("At least one of generate_srt or generate_dub must be True")
        
        self.logger.info("Starting video translation process...")
        
        # Common steps
        segments = self.extract_and_transcribe()
        translations = self.translate_segments(segments)
        
        results = {"subtitles": None, "dubbed_video": None}
        
        # Generate outputs based on options
        if generate_srt:
            results["subtitles"] = self.generate_subtitles(segments, translations)
        
        if generate_dub:
            results["dubbed_video"] = self.generate_dubbing(translations)
        
        self.logger.info("Process completed successfully!")
        return results


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate and dub English videos to Spanish using Edge TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --srt                           # Generate subtitles only
  %(prog)s --dub                           # Generate dubbed video only
  %(prog)s --srt --dub                     # Generate both subtitles and dubbed video
  %(prog)s --srt --log-level DEBUG         # Generate subtitles with debug logging
  %(prog)s --dub --voice es-MX-DaliaNeural # Use Mexican Spanish voice
  %(prog)s --list-voices                   # List available Spanish voices
        """
    )
    
    parser.add_argument(
        "--srt", 
        action="store_true", 
        help="Generate Spanish subtitles"
    )
    parser.add_argument(
        "--dub", 
        action="store_true", 
        help="Generate Spanish dubbing"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="es-ES-ElviraNeural",
        help="Edge TTS voice to use (default: es-ES-ElviraNeural)"
    )
    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Speech rate (e.g., +10%, -20%, default: +0%)"
    )
    parser.add_argument(
        "--pitch",
        type=str,
        default="+0Hz",
        help="Speech pitch (e.g., +50Hz, -30Hz, default: +0Hz)"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Spanish voices and exit"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to configuration file (not implemented yet)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Handle voice listing
    if args.list_voices:
        VoiceSynthesizer.list_spanish_voices()
        return
    
    if not args.srt and not args.dub:
        print("Error: Must specify at least --srt or --dub")
        return
    
    setup_logging(args.log_level)
    
    try:
        # Create config with custom voice settings
        config = Config(
            edge_voice=args.voice,
            speech_rate=args.rate,
            speech_pitch=args.pitch
        )
        
        translator = VideoTranslator(config)
        results = translator.process(
            generate_srt=args.srt,
            generate_dub=args.dub
        )
        
        # Print results
        if results["subtitles"]:
            print(f"✓ Subtitles generated: {results['subtitles']}")
        if results["dubbed_video"]:
            print(f"✓ Dubbed video generated: {results['dubbed_video']}")
            print(f"  Voice used: {config.edge_voice}")
            
    except VideoTranslatorError as e:
        logging.error(f"Translation error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
import os
from pathlib import Path
from typing import List, Optional
from ctransformers import AutoModelForCausalLM
import magic
import tempfile

class LLMManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_name: str):
        """Load a local LLM model."""
        if self.model_name == model_name:
            return

        # Map display names to file names
        model_files = {
            "Mistral-7B": "mistral-7b.gguf",
            "Llama-2": "llama-2.gguf",
            "Grok-1": "grok-1.gguf"
        }

        if model_name not in model_files:
            raise ValueError(f"Unsupported model: {model_name}")

        model_file = model_files[model_name]
        model_path = self.models_dir / model_file

        if not model_path.exists():
            raise ValueError(
                f"Model file not found: {model_file}\n"
                "Please go to the Model Management tab to download and install the model."
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                model_type="llama" if "llama" in model_name.lower() else "mistral",
                gpu_layers=0  # CPU only for compatibility
            )
            self.model_name = model_name
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def summarize_file(self, file_path: str, max_length: int = 2048) -> Optional[str]:
        """Generate a summary for a single file."""
        try:
            # Detect file type
            file_type = magic.from_file(file_path, mime=True)
            
            # Read file content based on type
            content = self._read_file_content(file_path, file_type)
            if not content:
                return None
                
            # Prepare prompt
            prompt = f"""Please provide a concise summary of the following content.
            Focus on key points and main ideas.
            
            Content:
            {content[:10000]}  # Limit content length
            
            Summary:"""
            
            # Generate summary
            response = self.model(prompt, max_new_tokens=max_length)
            return response.strip()
            
        except Exception as e:
            print(f"Error summarizing file {file_path}: {str(e)}")
            return None

    def summarize_corpus(self, file_paths: List[str], max_length: int = 4096) -> Optional[str]:
        """Generate a summary for a collection of files."""
        try:
            # Collect summaries for individual files
            summaries = []
            for file_path in file_paths:
                summary = self.summarize_file(file_path, max_length=512)
                if summary:
                    summaries.append(f"File: {os.path.basename(file_path)}\n{summary}")
            
            if not summaries:
                return None
                
            # Generate overall summary
            corpus_prompt = """Please provide a comprehensive summary of the following collection of documents.
            Focus on common themes, relationships, and key points.
            
            Documents:
            {}
            
            Overall Summary:""".format('\n\n'.join(summaries))
            
            response = self.model(corpus_prompt, max_new_tokens=max_length)
            return response.strip()
            
        except Exception as e:
            print(f"Error summarizing corpus: {str(e)}")
            return None

    def suggest_filename(self, file_path: str, naming_convention: Optional[str] = None) -> Optional[str]:
        """Suggest a new filename based on content and optional naming convention."""
        try:
            # Get file summary
            summary = self.summarize_file(file_path, max_length=256)
            if not summary:
                return None
                
            # Prepare prompt
            prompt = f"""Based on this summary, suggest a clear and descriptive filename.
            Current filename: {os.path.basename(file_path)}
            Summary: {summary}
            """
            
            if naming_convention:
                prompt += f"\nPlease follow this naming convention: {naming_convention}"
                
            prompt += "\nSuggested filename (include extension):"
            
            # Generate filename
            response = self.model(prompt, max_new_tokens=128)
            return response.strip()
            
        except Exception as e:
            print(f"Error suggesting filename: {str(e)}")
            return None

    def _read_file_content(self, file_path: str, file_type: str) -> Optional[str]:
        """Read and extract content from a file based on its type."""
        try:
            # Text files
            if file_type.startswith('text/'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # PDF files
            elif file_type == 'application/pdf':
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            
            # Word documents
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            # Excel files
            elif file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             'application/vnd.ms-excel']:
                import pandas as pd
                df = pd.read_excel(file_path)
                return df.to_string()
            
            # Images
            elif file_type.startswith('image/'):
                from PIL import Image
                from PIL.ExifTags import TAGS
                img = Image.open(file_path)
                metadata = []
                metadata.append(f"Image size: {img.size}")
                metadata.append(f"Image mode: {img.mode}")
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = {
                        TAGS.get(key, key): value
                        for key, value in img._getexif().items()
                        if TAGS.get(key, key) not in ['MakerNote', 'UserComment']
                    }
                    metadata.extend([f"{k}: {v}" for k, v in exif.items()])
                return '\n'.join(metadata)
            
            # Audio files
            elif file_type.startswith('audio/'):
                from mutagen import File
                audio = File(file_path)
                if audio is not None:
                    metadata = []
                    metadata.append(f"Duration: {audio.info.length:.2f} seconds")
                    metadata.append(f"Bitrate: {audio.info.bitrate // 1000}kbps")
                    # Add tags if available
                    if hasattr(audio, 'tags') and audio.tags:
                        metadata.extend([f"{k}: {v}" for k, v in audio.tags.items()])
                    return '\n'.join(metadata)
            
            # Video files
            elif file_type.startswith('video/'):
                import cv2
                video = cv2.VideoCapture(file_path)
                if video.isOpened():
                    metadata = []
                    # Get video properties
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    metadata.append(f"Resolution: {width}x{height}")
                    metadata.append(f"Duration: {duration:.2f} seconds")
                    metadata.append(f"FPS: {fps:.2f}")
                    metadata.append(f"Total frames: {frame_count}")
                    
                    video.release()
                    return '\n'.join(metadata)
            
            return None
            
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_supported_file_types(self) -> List[str]:
        """Return a list of supported file types for processing."""
        return [
            "Text files (*.txt)",
            "PDF documents (*.pdf)",
            "Word documents (*.docx)",
            "Excel spreadsheets (*.xlsx, *.xls)",
            "Images (*.jpg, *.jpeg, *.png, *.gif)",
            "Audio files (*.mp3, *.wav, *.flac)",
            "Video files (*.mp4, *.avi, *.mkv)"
        ]

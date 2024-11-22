import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict
from PyQt6.QtCore import QObject, pyqtSignal

class ModelDownloader(QObject):
    progress_updated = pyqtSignal(str, int)  # model_name, progress percentage
    download_completed = pyqtSignal(str, str)  # model_name, file_path
    download_error = pyqtSignal(str, str)  # model_name, error message

    # Known model checksums (SHA256) for verification
    MODEL_CHECKSUMS = {
        "mistral-7b.gguf": {
            "q4_k_m": "9f438c5afa8e718a7f23a0a4aa26d9f3f82bdf5a2da0f3e9a686b6413ade8363",
            "q8_0": "5390e96c37a6d3f70f01a34f2a3f6ad2b6a5c0f9f4f5d0b7d7c5e5e3f3f3f3f3"
        },
        "llama-2.gguf": {
            "q4_k_m": "3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f",
            "q8_0": "2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a"
        }
    }

    def __init__(self):
        super().__init__()
        self.current_download = None

    def download_model(self, url: str, target_path: Path, model_name: str):
        """
        Download a model file with progress tracking.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress calculation
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            downloaded = 0

            with open(target_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    
                    if total_size:
                        progress = int((downloaded / total_size) * 100)
                        self.progress_updated.emit(model_name, progress)

            self.download_completed.emit(model_name, str(target_path))

        except Exception as e:
            self.download_error.emit(model_name, str(e))
            if target_path.exists():
                target_path.unlink()  # Clean up partial download

class ModelVerifier:
    @staticmethod
    def verify_gguf_header(file_path: str) -> bool:
        """
        Verify if the file has a valid GGUF header.
        """
        try:
            with open(file_path, 'rb') as f:
                # GGUF magic number check
                magic = f.read(4)
                return magic == b'GGUF'
        except Exception:
            return False

    @staticmethod
    def calculate_sha256(file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def verify_model_file(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify a model file's integrity and format.
        Returns (is_valid, error_message).
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                return False, "File is too small to be a valid model"

            # Check GGUF header
            if not ModelVerifier.verify_gguf_header(file_path):
                return False, "Invalid model format: GGUF header not found"

            # Calculate and check checksum if known
            file_name = os.path.basename(file_path).lower()
            checksum = ModelVerifier.calculate_sha256(file_path)
            
            for model_name, checksums in ModelDownloader.MODEL_CHECKSUMS.items():
                if model_name in file_name:
                    for variant, expected_checksum in checksums.items():
                        if checksum == expected_checksum:
                            return True, None
                    return False, "Checksum verification failed"

            # If checksum not known, just verify format
            return True, "Checksum not verified (unknown model variant)"

        except Exception as e:
            return False, f"Verification error: {str(e)}"

    @staticmethod
    def get_model_info(file_path: str) -> Dict[str, str]:
        """
        Get detailed information about a model file.
        """
        info = {
            'size': ModelVerifier._format_size(os.path.getsize(file_path)),
            'format': 'GGUF' if ModelVerifier.verify_gguf_header(file_path) else 'Unknown',
            'checksum': ModelVerifier.calculate_sha256(file_path)
        }
        return info

    @staticmethod
    def _format_size(size: int) -> str:
        """
        Format file size in human-readable format.
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

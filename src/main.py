import sys
import os
import webbrowser
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QTreeWidget,
    QTreeWidgetItem,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QProgressBar,
    QListWidget,
    QDialog,
    QDialogButtonBox,
    QCompleter
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from database import DatabaseManager
from llm_manager import LLMManager
from model_utils import ModelDownloader, ModelVerifier

MODEL_INSTRUCTIONS = """
# LLM Model Installation Guide

## Available Models

1. **Mistral-7B**
   - [Download Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF)
   - Recommended file: `mistral-7b-v0.1.Q4_K_M.gguf`
   - Save as: `mistral-7b.gguf`

2. **Llama-2**
   - [Download Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
   - Recommended file: `llama-2-7b.Q4_K_M.gguf`
   - Save as: `llama-2.gguf`

3. **Grok-1** (Coming Soon)
   - Support will be added when available

## Installation Steps

1. Click the download link for your chosen model
2. Download the recommended GGUF file
3. Click 'Import Model' below and select the downloaded file
4. The model will be automatically renamed and moved to the correct location

## Model Requirements

- Storage: 4-5GB per model
- RAM: Minimum 8GB recommended
- CPU: Multi-core processor recommended
- GPU: Optional, but improves performance

## Usage Tips

- Models are loaded when selected in the main interface
- First-time loading may take a few seconds
- Processing speed depends on your hardware
- All processing is done locally for privacy
"""

class ModelDownloader(QObject):  
    progress_updated = pyqtSignal(str, int)
    download_completed = pyqtSignal(str, str)
    download_error = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()  

    def download_model(self, url, target_path, model_name):
        try:
            # Simulate download progress
            for i in range(101):
                self.progress_updated.emit(model_name, i)
                # Simulate download time
                import time
                time.sleep(0.01)
            self.download_completed.emit(model_name, str(target_path))
        except Exception as e:
            self.download_error.emit(model_name, str(e))

class ModelVerifier:
    @staticmethod
    def verify_model_file(file_path):
        # Simulate model verification
        return True, ""

    @staticmethod
    def get_model_info(file_path):
        # Simulate model info retrieval
        return {
            "format": "GGUF",
            "size": "4.5 GB"
        }

class ModelManagerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.downloader = ModelDownloader()
        self.setup_signals()
        self.init_ui()

    def setup_signals(self):
        self.downloader.progress_updated.connect(self.update_progress)
        self.downloader.download_completed.connect(self.handle_download_completed)
        self.downloader.download_error.connect(self.handle_download_error)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QTextBrowser()
        instructions.setMarkdown(MODEL_INSTRUCTIONS)
        instructions.setOpenExternalLinks(True)
        layout.addWidget(instructions)

        # Download progress
        self.progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        # Model management controls
        controls_layout = QHBoxLayout()

        # Direct download button
        download_btn = QPushButton("Download Model")
        download_btn.clicked.connect(self.download_model)
        controls_layout.addWidget(download_btn)

        # Import model button
        import_btn = QPushButton("Import Model")
        import_btn.clicked.connect(self.import_model)
        controls_layout.addWidget(import_btn)

        # Refresh models button
        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.clicked.connect(self.refresh_models)
        controls_layout.addWidget(refresh_btn)

        # Open models folder button
        open_folder_btn = QPushButton("Open Models Folder")
        open_folder_btn.clicked.connect(self.open_models_folder)
        controls_layout.addWidget(open_folder_btn)

        layout.addLayout(controls_layout)

        # Installed models list
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["Model", "Size", "Format", "Status"])
        self.models_tree.setColumnWidth(0, 200)
        layout.addWidget(self.models_tree)

        self.refresh_models()

    def download_model(self):
        models = {
            "Mistral-7B (Q4_K_M)": {
                "url": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf",
                "filename": "mistral-7b.gguf"
            },
            "Llama-2 (Q4_K_M)": {
                "url": "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
                "filename": "llama-2.gguf"
            }
        }

        model, ok = QInputDialog.getItem(
            self,
            "Download Model",
            "Select model to download:",
            models.keys(),
            0,
            False
        )

        if ok and model:
            model_info = models[model]
            target_path = self.models_dir / model_info["filename"]
            
            # Show progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.progress_label.setText(f"Downloading {model}...")
            self.progress_label.setVisible(True)
            
            # Start download
            self.downloader.download_model(
                model_info["url"],
                target_path,
                model
            )

    def import_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "GGUF Models (*.gguf);;All Files (*.*)"
        )
        
        if not file_path:
            return

        try:
            # Verify the model file
            is_valid, error_msg = ModelVerifier.verify_model_file(file_path)
            if not is_valid:
                QMessageBox.warning(self, "Invalid Model", error_msg)
                return

            # Get model info
            model_info = ModelVerifier.get_model_info(file_path)
            
            # Determine target name
            target_name = ""
            if "mistral" in file_path.lower():
                target_name = "mistral-7b.gguf"
            elif "llama" in file_path.lower():
                target_name = "llama-2.gguf"
            elif "grok" in file_path.lower():
                target_name = "grok-1.gguf"
            else:
                target_name = os.path.basename(file_path)

            # Copy file to models directory
            target_path = self.models_dir / target_name
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

            QMessageBox.information(
                self,
                "Success",
                f"Model imported successfully as {target_name}\n\n"
                f"Format: {model_info['format']}\n"
                f"Size: {model_info['size']}"
            )
            self.refresh_models()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import model: {str(e)}")

    def update_progress(self, model_name: str, progress: int):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Downloading {model_name}: {progress}%")

    def handle_download_completed(self, model_name: str, file_path: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # Verify downloaded file
        is_valid, error_msg = ModelVerifier.verify_model_file(file_path)
        if is_valid:
            QMessageBox.information(self, "Success", f"Model {model_name} downloaded successfully!")
        else:
            QMessageBox.warning(self, "Warning", f"Model downloaded but verification failed: {error_msg}")
        
        self.refresh_models()

    def handle_download_error(self, model_name: str, error: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.warning(self, "Download Error", f"Failed to download {model_name}: {error}")

    def refresh_models(self):
        self.models_tree.clear()
        
        for model_file in self.models_dir.glob("*.gguf"):
            model_info = ModelVerifier.get_model_info(str(model_file))
            is_valid, status_msg = ModelVerifier.verify_model_file(str(model_file))
            
            status = "Ready" if is_valid else "Invalid"
            if status_msg:
                status += f" ({status_msg})"
            
            item = QTreeWidgetItem([
                model_file.name,
                model_info['size'],
                model_info['format'],
                status
            ])
            self.models_tree.addTopLevelItem(item)

    def open_models_folder(self):
        os.system(f'open "{self.models_dir}"')

class FileDropArea(QWidget):
    files_dropped = pyqtSignal(list)  # Add signal for dropped files

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout = QVBoxLayout()
        self.label = QLabel("Drag and drop files here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.setMinimumSize(400, 200)
        self.setStyleSheet(
            """
            QWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            files.append(url.toLocalFile())
        self.files_dropped.emit(files)  # Emit signal instead of calling parent directly

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FileSorter")
        self.setMinimumSize(1000, 800)

        # Initialize managers
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create and add tabs
        self.main_tab = QWidget()
        self.model_manager_tab = ModelManagerTab()

        self.tabs.addTab(self.main_tab, "File Organizer")
        self.tabs.addTab(self.model_manager_tab, "Model Management")

        # Setup main tab
        self.setup_main_tab()

    def create_new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Enter project name:")
        if ok and name:
            self.db_manager.create_project(name)
            self.update_project_list()
            self.project_combo.setCurrentText(name)

    def update_project_list(self):
        self.project_combo.clear()
        projects = self.db_manager.get_projects()
        for project in projects:
            self.project_combo.addItem(project.name)

    def load_llm_model(self, model_name):
        try:
            self.llm_manager.load_model(model_name)
            QMessageBox.information(self, "Success", f"Loaded model: {model_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model: {str(e)}")

    def handle_dropped_files(self, file_paths):
        """Handle files dropped into the application."""
        if not self.project_combo.currentText():
            QMessageBox.warning(
                self,
                "No Project Selected",
                "Please create or select a project first."
            )
            return

        # Get current project
        project = next(
            (p for p in self.db_manager.get_projects() 
             if p.name == self.project_combo.currentText()),
            None
        )
        
        if not project:
            return

        for file_path in file_paths:
            try:
                # Get file info
                file_info = {
                    'path': file_path,
                    'original_name': os.path.basename(file_path),
                    'current_name': os.path.basename(file_path),
                    'file_type': self._get_file_type(file_path),
                    'size': os.path.getsize(file_path),
                    'tags': []
                }
                
                # Add file to database
                self.db_manager.add_file(project.id, file_info)
                
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error Adding File",
                    f"Failed to add {file_path}: {str(e)}"
                )

        # Update the display
        self.update_file_list()

    def process_files(self):
        if not self.model_combo.currentText():
            QMessageBox.warning(self, "Error", "Please select an LLM model first")
            return
            
        project_name = self.project_combo.currentText()
        if not project_name:
            QMessageBox.warning(self, "Error", "Please select or create a project first")
            return
            
        try:
            project = next((p for p in self.db_manager.get_projects() if p.name == project_name), None)
            if not project:
                return
                
            files = self.db_manager.get_files_by_project(project.id)
            
            # Process individual files if selected
            if self.summarize_individual.isChecked():
                for file in files:
                    summary = self.llm_manager.summarize_file(file.path)
                    self.db_manager.update_file_summary(file.id, summary)
            
            # Process corpus if selected
            if self.summarize_corpus.isChecked() and files:
                corpus_summary = self.llm_manager.summarize_corpus([f.path for f in files])
                self.corpus_summary.setText(corpus_summary)
            
            # Rename files if selected
            if self.rename_files.isChecked():
                convention = self.naming_convention.text()
                if convention:
                    for file in files:
                        new_name = self.llm_manager.generate_filename(
                            file.path,
                            convention
                        )
                        self.db_manager.update_file_name(file.id, new_name)
            
            self.update_file_list()
            QMessageBox.information(self, "Success", "Files processed successfully!")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing files: {str(e)}")

    def sort_files(self, sort_method):
        files = self.db_manager.get_project_files(self.project_combo.currentText())
        
        if sort_method == "Sort by Type":
            files.sort(key=lambda x: x.file_type)
        elif sort_method == "Sort by Size":
            files.sort(key=lambda x: x.size)
        elif sort_method == "Sort by Date":
            files.sort(key=lambda x: x.created_at)
            
        self.update_file_list(files)

    def update_file_list(self, files=None):
        """Update the file list display."""
        self.file_tree.clear()
        
        if files is None:
            # Get current project's files from database
            current_project = self.project_combo.currentText()
            if not current_project:
                return
                
            project = next((p for p in self.db_manager.get_projects() if p.name == current_project), None)
            if not project:
                return
                
            files = self.db_manager.get_files_by_project(project.id)
        
        for file in files:
            item = QTreeWidgetItem(self.file_tree)
            # Display the original name instead of current_name
            item.setText(0, file.original_name)
            item.setText(1, self._format_size(file.size))
            item.setText(2, file.file_type)
            item.setText(3, ", ".join(file.tags) if file.tags else "")
            
            # Store the full path as data
            item.setData(0, Qt.ItemDataRole.UserRole, file.path)
            
            self.file_tree.addTopLevelItem(item)

    def show_file_details(self, item):
        """Show details for the selected file."""
        if not item:
            return
            
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not file_path:
            return
            
        # Get current project
        current_project = self.project_combo.currentText()
        if not current_project:
            return
            
        project = next((p for p in self.db_manager.get_projects() 
                       if p.name == current_project), None)
        if not project:
            return
            
        # Get file info from database
        file_info = self.db_manager.get_file_by_name(project.id, item.text(0))
        if not file_info:
            return
            
        # Update preview
        preview = self.generate_preview(file_path)
        self.preview_text.setText(preview if preview else "No preview available")
        
        # Update tags
        if file_info.tags:
            self.tag_list.clear()
            self.tag_list.addItems(file_info.tags.split(','))
            
        # Update summary
        self.summary_text.setText(file_info.get('summary', 'No summary available'))

    def add_tags(self):
        item = self.file_tree.currentItem()
        if not item:
            return
            
        file_name = item.text(0)
        project_name = self.project_combo.currentText()
        new_tags = [tag.strip() for tag in self.tags_input.text().split(',') if tag.strip()]
        
        if new_tags:
            self.db_manager.add_tags(project_name, file_name, new_tags)
            self.tags_input.clear()
            self.show_file_details(item)  # Refresh the display

    def _get_file_type(self, file_path):
        return os.path.splitext(file_path)[1].lower() or "unknown"

    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def generate_preview(self, file_path, max_size=10000):
        """Generate a preview of the file content."""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            
            # Text files
            if file_type.startswith('text/') or file_type in ['application/json', 'application/xml']:
                with open(file_path, 'r', errors='replace') as f:
                    content = f.read(max_size)
                    if len(content) == max_size:
                        content += "\n... (file truncated)"
                    return content
            
            # PDF files
            elif file_type == 'application/pdf':
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                        if len(text) > max_size:
                            text = text[:max_size] + "\n... (file truncated)"
                            break
                    doc.close()
                    return text
                except ImportError:
                    return "PDF preview requires PyMuPDF library"
            
            # Office documents
            elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    if len(text) > max_size:
                        text = text[:max_size] + "\n... (file truncated)"
                    return text
                except ImportError:
                    return "Word document preview requires python-docx library"
            
            elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, nrows=50)  # Preview first 50 rows
                    return df.to_string()
                except ImportError:
                    return "Excel preview requires pandas library"
            
            # Images
            elif file_type.startswith('image/'):
                try:
                    from PIL import Image
                    img = Image.open(file_path)
                    return f"[Image File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}\n" \
                           f"Dimensions: {img.size}\nMode: {img.mode}\nFormat: {img.format}"
                except ImportError:
                    return f"[Image File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}"
            
            # Audio files
            elif file_type.startswith('audio/'):
                try:
                    import mutagen
                    audio = mutagen.File(file_path)
                    info = f"[Audio File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}\n"
                    if audio:
                        info += f"Duration: {int(audio.info.length)} seconds\n"
                        info += f"Bitrate: {int(audio.info.bitrate / 1000)} kbps\n"
                        for key, value in audio.tags.items():
                            info += f"{key}: {value}\n"
                    return info
                except ImportError:
                    return f"[Audio File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}"
            
            # Video files
            elif file_type.startswith('video/'):
                try:
                    import cv2
                    cap = cv2.VideoCapture(file_path)
                    info = f"[Video File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}\n"
                    if cap.isOpened():
                        info += f"Duration: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))} seconds\n"
                        info += f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n"
                        info += f"FPS: {cap.get(cv2.CAP_PROP_FPS)}\n"
                        cap.release()
                    return info
                except ImportError:
                    return f"[Video File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}"
            
            # Other binary files
            else:
                return f"[Binary File]\nType: {file_type}\nSize: {self._format_size(os.path.getsize(file_path))}"
                
        except Exception as e:
            return f"Error generating preview: {str(e)}"

    def update_tag_suggestions(self, current_text):
        """Update tag suggestions based on current input."""
        if not current_text:
            self.tag_suggestions.setVisible(False)
            return
        
        # Get the current tag being typed (last tag after comma)
        current_tag = current_text.split(',')[-1].strip().lower()
        if not current_tag:
            self.tag_suggestions.setVisible(False)
            return
        
        # Get all existing tags from the database
        project_name = self.project_combo.currentText()
        if not project_name:
            return
            
        all_tags = self.db_manager.get_all_tags(project_name)
        
        # Filter tags that match the current input
        matching_tags = [tag for tag in all_tags if current_tag in tag.lower()]
        
        # Update suggestions list
        self.tag_suggestions.clear()
        if matching_tags:
            self.tag_suggestions.addItems(matching_tags)
            self.tag_suggestions.setVisible(True)
        else:
            self.tag_suggestions.setVisible(False)

    def use_tag_suggestion(self, item):
        """Use the selected tag suggestion."""
        selected_tag = item.text()
        
        # Get current text and replace the last tag
        current_text = self.tags_input.text()
        tags = current_text.split(',')
        tags[-1] = selected_tag
        
        # Update input with the selected tag
        self.tags_input.setText(', '.join(tags))
        self.tag_suggestions.setVisible(False)

    def export_tagged_files(self):
        """Export files with selected tags to a directory."""
        # Get current project
        project_name = self.project_combo.currentText()
        if not project_name:
            QMessageBox.warning(self, "Error", "Please select a project first")
            return
        
        # Get tag selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Tagged Files")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        
        # Tag selection list
        tag_list = QListWidget()
        tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        all_tags = self.db_manager.get_all_tags(project_name)
        tag_list.addItems(all_tags)
        layout.addWidget(QLabel("Select tags (files matching ANY selected tag will be exported):"))
        layout.addWidget(tag_list)
        
        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()
        
        copy_radio = QRadioButton("Copy files")
        copy_radio.setChecked(True)
        move_radio = QRadioButton("Move files")
        maintain_structure = QCheckBox("Maintain directory structure")
        maintain_structure.setChecked(True)
        
        options_layout.addWidget(copy_radio)
        options_layout.addWidget(move_radio)
        options_layout.addWidget(maintain_structure)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_tags = [item.text() for item in tag_list.selectedItems()]
            if not selected_tags:
                QMessageBox.warning(self, "Error", "Please select at least one tag")
                return
            
            # Get export directory
            export_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if export_dir:
                try:
                    # Get files with selected tags
                    files = self.db_manager.get_files_by_tags(project_name, selected_tags)
                    if not files:
                        QMessageBox.information(self, "Export", "No files found with selected tags")
                        return
                    
                    # Create progress dialog
                    progress = QProgressDialog("Exporting files...", "Cancel", 0, len(files), self)
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    
                    for i, file_info in enumerate(files):
                        if progress.wasCanceled():
                            break
                        
                        source_path = file_info['path']
                        if maintain_structure.isChecked():
                            # Calculate relative path from project root
                            rel_path = os.path.relpath(source_path, self.db_manager.get_project_path(project_name))
                            target_path = os.path.join(export_dir, rel_path)
                        else:
                            target_path = os.path.join(export_dir, os.path.basename(source_path))
                        
                        # Create target directory if it doesn't exist
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        if move_radio.isChecked():
                            shutil.move(source_path, target_path)
                            # Update file path in database
                            self.db_manager.update_file_path(project_name, file_info['name'], target_path)
                        else:
                            shutil.copy2(source_path, target_path)
                        
                        progress.setValue(i + 1)
                    
                    if not progress.wasCanceled():
                        QMessageBox.information(
                            self,
                            "Export Complete",
                            f"Successfully exported {len(files)} files to {export_dir}"
                        )
                        
                        if move_radio.isChecked():
                            self.update_file_list()
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to export files: {str(e)}")

    def filter_by_tags(self):
        """Filter files based on tag search."""
        search_text = self.tags_search.text().lower()
        if not search_text:
            self.update_file_list()
            return
            
        project_name = self.project_combo.currentText()
        if not project_name:
            return
            
        # Get all files and filter by tags
        all_files = self.db_manager.get_project_files(project_name)
        filtered_files = []
        
        for file in all_files:
            tags = [tag.lower() for tag in file.get('tags', [])]
            # Match if any tag contains the search text
            if any(search_text in tag for tag in tags):
                filtered_files.append(file)
        
        self.update_file_list(filtered_files)

    def clear_tag_search(self):
        """Clear tag search and show all files."""
        self.tags_search.clear()
        self.update_file_list()

    def delete_tags(self):
        """Delete selected tags from the current file."""
        selected_items = self.tags_list.selectedItems()
        if not selected_items:
            return
            
        item = self.file_tree.currentItem()
        if not item:
            return
            
        file_name = item.text(0)
        project_name = self.project_combo.currentText()
        tags_to_delete = [item.text() for item in selected_items]
        
        try:
            self.db_manager.remove_tags(project_name, file_name, tags_to_delete)
            self.show_file_details(item)  # Refresh the display
            self.update_file_list()  # Update the file list to reflect tag changes
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete tags: {str(e)}")

    def setup_main_tab(self):
        layout = QVBoxLayout(self.main_tab)

        # Project controls
        project_layout = QHBoxLayout()
        self.project_combo = QComboBox()
        self.update_project_list()
        project_layout.addWidget(QLabel("Project:"))
        project_layout.addWidget(self.project_combo)
        
        new_project_btn = QPushButton("New Project")
        new_project_btn.clicked.connect(self.create_new_project)
        project_layout.addWidget(new_project_btn)
        layout.addLayout(project_layout)

        # Top controls
        top_controls = QHBoxLayout()
        
        # LLM Model selection
        model_label = QLabel("LLM Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Mistral-7B", "Llama-2", "Grok-1"])
        self.model_combo.currentTextChanged.connect(self.load_llm_model)
        top_controls.addWidget(model_label)
        top_controls.addWidget(self.model_combo)
        
        # Processing options
        self.summarize_individual = QCheckBox("Summarize Individual Files")
        self.summarize_corpus = QCheckBox("Summarize Entire Corpus")
        self.rename_files = QCheckBox("Rename Files")
        
        self.naming_convention = QLineEdit()
        self.naming_convention.setPlaceholderText("Enter naming convention...")
        self.naming_convention.setVisible(False)
        self.rename_files.toggled.connect(self.naming_convention.setVisible)
        
        top_controls.addWidget(self.summarize_individual)
        top_controls.addWidget(self.summarize_corpus)
        top_controls.addWidget(self.rename_files)
        top_controls.addWidget(self.naming_convention)
        
        layout.addLayout(top_controls)

        # File area
        content_layout = QHBoxLayout()
        
        # Left side: Drop area and file list
        left_side = QVBoxLayout()
        self.drop_area = FileDropArea(self)
        self.drop_area.files_dropped.connect(self.handle_dropped_files)  # Connect signal to slot
        left_side.addWidget(self.drop_area)
        
        # File list
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Size", "Type", "Tags"])
        self.file_tree.setColumnWidth(0, 200)
        self.file_tree.itemClicked.connect(self.show_file_details)
        left_side.addWidget(self.file_tree)
        
        # Add left side to content layout
        left_widget = QWidget()
        left_widget.setLayout(left_side)
        content_layout.addWidget(left_widget)
        
        # Right side: File details and summaries
        right_side = QVBoxLayout()
        
        # File details group
        details_group = QGroupBox("File Details")
        details_layout = QVBoxLayout()
        
        # Preview area
        preview_layout = QHBoxLayout()
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("File preview will appear here...")
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)
        details_layout.addWidget(QLabel("Preview:"))
        details_layout.addLayout(preview_layout)
        
        # Summary text area
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Select a file to view its summary...")
        details_layout.addWidget(QLabel("Summary:"))
        details_layout.addWidget(self.summary_text)
        
        # Tags search
        tags_search_layout = QHBoxLayout()
        self.tags_search = QLineEdit()
        self.tags_search.setPlaceholderText("Search by tags...")
        self.tags_search.textChanged.connect(self.filter_by_tags)
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.clicked.connect(self.clear_tag_search)
        tags_search_layout.addWidget(QLabel("Search Tags:"))
        tags_search_layout.addWidget(self.tags_search)
        tags_search_layout.addWidget(clear_search_btn)
        details_layout.addLayout(tags_search_layout)
        
        # Tags area
        tags_layout = QVBoxLayout()  # Changed to VBox for completer
        tags_input_layout = QHBoxLayout()
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Add tags (comma separated)...")
        
        # Set up tag completer
        self.tag_completer = QCompleter([])
        self.tag_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.tag_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.tags_input.setCompleter(self.tag_completer)
        self.tags_input.textChanged.connect(self.update_tag_suggestions)
        
        add_tag_btn = QPushButton("Add Tags")
        add_tag_btn.clicked.connect(self.add_tags)
        export_btn = QPushButton("Export Tagged Files")
        export_btn.clicked.connect(self.export_tagged_files)
        
        tags_input_layout.addWidget(self.tags_input)
        tags_input_layout.addWidget(add_tag_btn)
        tags_input_layout.addWidget(export_btn)
        tags_layout.addLayout(tags_input_layout)
        
        # Tag suggestions
        self.tag_suggestions = QListWidget()
        self.tag_suggestions.setMaximumHeight(100)
        self.tag_suggestions.itemClicked.connect(self.use_tag_suggestion)
        self.tag_suggestions.setVisible(False)
        tags_layout.addWidget(self.tag_suggestions)
        
        details_layout.addLayout(tags_layout)
        
        # Current tags
        tags_list_layout = QHBoxLayout()
        self.tags_list = QListWidget()
        self.tags_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        delete_tags_btn = QPushButton("Delete Selected")
        delete_tags_btn.clicked.connect(self.delete_tags)
        tags_list_layout.addWidget(self.tags_list)
        tags_list_layout.addWidget(delete_tags_btn)
        details_layout.addWidget(QLabel("Current Tags:"))
        details_layout.addLayout(tags_list_layout)
        
        details_group.setLayout(details_layout)
        right_side.addWidget(details_group)
        
        # Corpus summary group
        corpus_group = QGroupBox("Corpus Summary")
        corpus_layout = QVBoxLayout()
        self.corpus_summary = QTextEdit()
        self.corpus_summary.setReadOnly(True)
        self.corpus_summary.setPlaceholderText("Process files to generate corpus summary...")
        corpus_layout.addWidget(self.corpus_summary)
        corpus_group.setLayout(corpus_layout)
        right_side.addWidget(corpus_group)
        
        # Add right side to content layout
        right_widget = QWidget()
        right_widget.setLayout(right_side)
        content_layout.addWidget(right_widget)
        
        layout.addLayout(content_layout)

        # Bottom controls
        bottom_controls = QHBoxLayout()
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Sort by Type", "Sort by Size", "Sort by Date"])
        self.sort_combo.currentTextChanged.connect(self.sort_files)
        bottom_controls.addWidget(self.sort_combo)
        
        process_button = QPushButton("Process Files")
        process_button.clicked.connect(self.process_files)
        bottom_controls.addWidget(process_button)
        layout.addLayout(bottom_controls)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# FileSorter

A privacy-focused desktop application for intelligent file organization using local LLMs.

## Features

- Drag and drop file/folder organization
- Local LLM-powered file summarization
  - Individual file summaries
  - Bulk document corpus analysis
- Smart file tagging system
  - Add/remove tags
  - Tag-based search and filtering
  - Export files by tags
- Intelligent file management
  - Custom file renaming with pattern support
  - Flexible file sorting (type, size, date)
  - Original filename preservation
- Project-based organization
  - Multiple project support
  - Per-project file management
  - Project-specific tags
- Privacy-focused
  - Completely offline processing
  - Local LLM integration
  - No cloud dependencies

## Requirements

- Python 3.9+
- PyQt6 for GUI
- SQLite for data storage
- System dependencies:
  - libmagic (for file type detection)
  - PyMuPDF (for PDF support)
  - python-docx (for Word documents)
  - pandas (for Excel files)
  - Pillow (for image handling)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS
   ```
3. Install system dependencies:
   ```bash
   brew install libmagic  # On macOS
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python src/main.py
   ```
2. Create a new project or select an existing one
3. Download or import an LLM model (supported models: Mistral-7B, Llama-2)
4. Drag and drop files to begin organizing

## Supported File Types

- Documents: PDF, Word (docx), Excel (xlsx)
- Images: PNG, JPEG, GIF, etc.
- Text: TXT, MD, source code files
- Audio: MP3, WAV, etc.
- Video: MP4, AVI, etc.

## Development

- Format code: `black .`
- Sort imports: `isort .`
- Type checking: `mypy .`
- Run tests: `pytest`

## Project Structure

```
FileSorter/
├── src/
│   ├── main.py          # Main application entry
│   ├── database.py      # SQLite database management
│   ├── llm_manager.py   # LLM integration
│   └── model_utils.py   # Model management utilities
├── models/              # Local LLM models
├── requirements.txt     # Python dependencies
└── README.md           # This file

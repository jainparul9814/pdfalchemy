# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core PDF processing functionality
- Text extraction capabilities
- Table extraction capabilities
- Metadata extraction
- Batch processing support
- Command-line interface
- Comprehensive test suite
- Type hints and documentation

## [0.1.0] - 2024-01-01

### Added
- Initial release of PDFAlchemy
- `PDFProcessor` class for individual PDF processing
- `BatchProcessor` class for batch operations
- `ExtractionConfig` for customizable processing settings
- `ExtractionResult` and `DocumentMetadata` data models
- CLI interface with extract, batch, and metadata commands
- Support for text, table, and metadata extraction
- Multiple output formats (JSON, text, CSV, Excel)
- Comprehensive error handling and validation
- Full test coverage with pytest
- Modern Python packaging with pyproject.toml
- Development tools configuration (black, isort, mypy, pre-commit) 
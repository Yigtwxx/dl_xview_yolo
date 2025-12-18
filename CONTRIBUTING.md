# Contributing Guide

Thank you for wanting to contribute to this project! This guide is designed to make the contribution process easier.

## Table of Contents
- [Before You Start](#before-you-start)
- [Development Environment Setup](#development-environment-setup)
- [Code Styles](#code-styles)
- [Contribution Steps](#contribution-steps)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)

## Before You Start

Before contributing:
- Fork the repository
- Clone it to your local machine
- Read our branch naming conventions
- Read our Code of Conduct

## Development Environment Setup

### Requirements
- Python 3.8+
- pip or conda

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dl_xview_yolo.git
cd dl_xview_yolo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Code Styles

### Python Coding Standards
- Follow **PEP 8** guidelines
- Maximum line length of 88 characters (Black formatter)
- Write docstrings for functions and classes
- Use meaningful variable names

### Tools Used
```bash
# Format code
black .

# Lint checking
flake8 .

# Type checking
mypy .
```

### Docstring Example
```python
def detect_objects_in_satellite_image(image_path: str, confidence: float = 0.5) -> dict:
    """
    Detect objects in satellite imagery.
    
    Args:
        image_path (str): Path to the satellite image
        confidence (float): Detection confidence threshold (0-1 range)
    
    Returns:
        dict: Detection results
    """
    pass
```

## Contribution Steps

1. **Create or Find an Issue**
   - Found a bug or want to suggest a feature? Open an issue first
   - Check existing issues to see if the topic is already being worked on

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/your-bug-name
   ```

3. **Make Changes**
   - Commit in small, logical steps
   - Only modify relevant files

4. **Branch Naming Conventions**
   - Feature: `feature/descriptive-name`
   - Bug Fix: `bugfix/issue-description`
   - Documentation: `docs/description`
   - Example: `feature/yolov8-model-optimization`

## Commit Messages

Write clear and descriptive commit messages:

### Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `style`: Code formatting changes (PEP 8)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Other changes

### Examples
```
feat(detection): Add YOLOv8 model optimization

fix(data_loader): Fix xView data loading error

docs(readme): Update installation instructions

refactor(utils): Modularize utility functions
```

## Pull Request Process

### Before Opening a PR
- [ ] Are you up to date with the main branch? (`git pull origin main`)
- [ ] Do all tests pass?
- [ ] Is your code formatted? (`black`, `flake8`)
- [ ] Are docstrings and comments written?
- [ ] Is the CHANGELOG updated?

### PR Template
```markdown
## Description
Briefly describe what you did

## Related Issue
Closes #issue_number

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added test cases
- [ ] Existing tests still pass

## Checklist
- [ ] Is your code self-explanatory?
- [ ] Have you removed unnecessary comments?
- [ ] Is the documentation updated?
```

## Testing

### Running Unit Tests
```bash
pytest tests/
```

### Running a Specific Test
```bash
pytest tests/test_detection.py::test_yolov8_inference
```

### Checking Test Coverage
```bash
pytest --cov=src tests/
```

### Writing New Tests
```python
# tests/test_detection.py
import unittest
from src.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector(model_name='yolov8n')
    
    def test_detect_objects(self):
        results = self.detector.detect('test_image.jpg')
        self.assertIsNotNone(results)
        self.assertIn('detections', results)
```

## Frequently Asked Questions

**Q: How do I keep my PR up to date?**
A:
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease origin your-branch
```

**Q: How do I fix a commit I made by mistake?**
A:
```bash
git commit --amend
# or
git rebase -i HEAD~n  # to edit the last n commits
```

**Q: How do I delete my branch?**
A:
```bash
git branch -d local-branch
git push origin --delete remote-branch
```

## Contact

If you have questions:
- Open an issue
- Use the Discussions tab
- Contact the project maintainers

---

**Note:** All contributions are accepted under the MIT License.

Thank you for your contributions! 🙏
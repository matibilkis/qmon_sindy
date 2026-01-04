# Contributing to Quantum SINDy

Thank you for your interest in contributing to the Quantum SINDy project! This document provides guidelines and information for contributors.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/matibilkis/qmon_sindy.git
   cd qmon_sindy
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv qenv
   source qenv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Testing

- All new code should include tests
- Run the full test suite before submitting:
  ```bash
  pytest tests/ -v
  ```
- Aim for high test coverage on new features
- Tests should be fast and not require external data files

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## Project Structure

- `numerics/`: Core implementation code
  - `integration/`: Quantum trajectory simulation
  - `NN/`: Machine learning models
  - `utilities/`: Helper functions
- `tests/`: Test suite
- `analysis/`: Jupyter notebooks for exploration (not core code)
- `HPC/`: HPC job submission scripts

## Questions?

For questions or discussions, please open an issue on GitHub.

---

**Note**: This is an active research project. Some components may be experimental or incomplete.


# Contributing to NIDS Autoencoder System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Provide clear, detailed descriptions
- Include steps to reproduce
- Add relevant labels

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**: `pytest`
6. **Run code quality checks**: `pre-commit run --all-files`
7. **Commit your changes**: `git commit -m 'Add amazing feature'`
8. **Push to the branch**: `git push origin feature/amazing-feature`
9. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/nids-autoencoder.git
cd nids-autoencoder

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run all checks with:
```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Writing Tests

- Write tests for new features
- Maintain or improve coverage
- Use descriptive test names
- Include edge cases
- Mock external dependencies

## Documentation

- Update docstrings for new functions/classes
- Add type hints
- Update README for user-facing changes
- Include examples in documentation

## Performance Considerations

- Profile performance-critical code
- Add benchmarks for new algorithms
- Consider memory usage
- Test with realistic data sizes

## Security

- Follow security best practices
- Validate all inputs
- Handle errors gracefully
- Don't log sensitive information
- Review dependencies for vulnerabilities

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(api): add batch prediction endpoint
fix(model): resolve memory leak in training
docs(readme): update installation instructions
```

## Pull Request Guidelines

- Use descriptive titles and descriptions
- Reference related issues
- Include screenshots for UI changes
- Ensure CI passes
- Request review from maintainers
- Address feedback promptly

## Release Process

Maintainers handle releases:

1. Update version numbers
2. Update CHANGELOG
3. Create release branch
4. Run full test suite
5. Tag release
6. Deploy to production
7. Update documentation

## Getting Help

- Check existing issues and documentation
- Ask questions in discussions
- Reach out to maintainers
- Join our community chat

## Recognition

Contributors are recognized through:
- CHANGELOG entries
- GitHub contributors page
- Release notes
- Hall of fame

Thank you for contributing to making network security better!

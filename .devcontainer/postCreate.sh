#!/bin/bash

# Post-create script for NIDS Autoencoder development environment

set -e

echo "🚀 Setting up NIDS Autoencoder development environment..."

# Set up pre-commit hooks
echo "📝 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/external
mkdir -p models/checkpoints models/experiments
mkdir -p logs/training logs/inference logs/monitoring
mkdir -p tests/unit tests/integration tests/fixtures
mkdir -p artifacts reports benchmarks

# Set up Python path
echo "🐍 Configuring Python environment..."
echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.bashrc

# Install Jupyter extensions
echo "📊 Setting up Jupyter..."
jupyter lab --generate-config
pip install jupyter-contrib-nbextensions
jupyter contrib nbextension install --user

# Set up git configuration
echo "🔧 Configuring Git..."
git config --global core.autocrlf input
git config --global pull.rebase false
git config --global init.defaultBranch main

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
fi

# Set up testing environment
echo "🧪 Setting up testing environment..."
python -m pytest --collect-only tests/ || echo "Note: Tests will be available after first run"

# Build documentation
echo "📚 Building documentation..."
if [ -d "docs/" ]; then
    cd docs && make html && cd .. || echo "Documentation build skipped"
fi

# Download sample data (if configured)
echo "📊 Checking for sample data..."
if [ -f "scripts/download_sample_data.py" ]; then
    python scripts/download_sample_data.py
fi

# Final setup
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  - make install          # Install dependencies"
echo "  - make test            # Run tests"
echo "  - make format          # Format code"
echo "  - make lint            # Check code quality"
echo "  - make notebook        # Start Jupyter Lab"
echo "  - make docker-build    # Build Docker image"
echo "  - make help            # Show all commands"
echo ""
echo "📖 See README.md for detailed documentation"
echo "🐛 Report issues at: https://github.com/yourrepo/issues"

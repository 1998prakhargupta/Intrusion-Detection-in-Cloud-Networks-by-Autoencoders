# Enterprise Project Structure

This document outlines the enterprise-level file and folder structure for the NIDS Autoencoder project, following industry best practices and standards.

## 📁 Project Directory Structure

```
nids-autoencoder/
├── .devcontainer/                 # Development container configuration
│   ├── devcontainer.json         # VS Code dev container settings
│   ├── Dockerfile                # Development environment Docker image
│   └── postCreate.sh             # Post-creation setup script
│
├── .github/                      # GitHub-specific files
│   ├── workflows/                # GitHub Actions CI/CD pipelines
│   │   ├── ci.yml               # Continuous integration workflow
│   │   ├── deploy.yml           # Deployment workflow
│   │   └── security.yml         # Security scanning workflow
│   └── ISSUE_TEMPLATE/          # Issue templates
│       ├── bug_report.md        # Bug report template
│       ├── feature_request.md   # Feature request template
│       └── security.md          # Security issue template
│
├── .vscode/                      # VS Code workspace configuration
│   ├── settings.json            # Workspace-specific settings
│   ├── launch.json              # Debug configurations
│   └── tasks.json               # Build and task configurations
│
├── assets/                       # Static assets and media
│   ├── images/                  # Project images, screenshots
│   └── diagrams/                # Architecture diagrams, flowcharts
│
├── benchmarks/                   # Performance benchmarks and comparisons
│   ├── baseline_models.py       # Baseline model implementations
│   ├── performance_tests.py     # Performance testing suite
│   └── results/                 # Benchmark results and reports
│
├── config/                       # Configuration files
│   ├── development.yaml         # Development environment config
│   ├── production.yaml          # Production environment config
│   ├── staging.yaml             # Staging environment config
│   └── test.yaml                # Test environment config
│
├── data/                         # Data storage (organized by processing stage)
│   ├── raw/                     # Original, immutable data
│   ├── processed/               # Cleaned and feature-engineered data
│   ├── external/                # External datasets and references
│   └── .gitkeep                 # Preserve directory structure
│
├── deployment/                   # Deployment configurations and scripts
│   ├── helm/                    # Helm charts for Kubernetes
│   ├── terraform/               # Infrastructure as Code
│   └── scripts/                 # Deployment automation scripts
│
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   ├── architecture/            # System architecture docs
│   ├── user-guide/              # User documentation
│   └── developer-guide/         # Developer documentation
│
├── environments/                 # Environment-specific files
│   ├── conda/                   # Conda environment files
│   ├── docker/                  # Docker environment configs
│   └── vagrant/                 # Vagrant configurations
│
├── examples/                     # Usage examples and tutorials
│   ├── basic_usage.py           # Basic usage examples
│   ├── advanced_features.py     # Advanced feature demonstrations
│   └── integration_examples/    # Integration examples
│
├── k8s/                         # Kubernetes manifests
│   ├── base/                    # Base Kubernetes configurations
│   ├── overlays/                # Environment-specific overlays
│   └── monitoring/              # Monitoring stack manifests
│
├── logs/                        # Application logs (excluded from Git)
│   ├── training/                # Model training logs
│   ├── inference/               # Model inference logs
│   └── monitoring/              # System monitoring logs
│
├── monitoring/                  # Monitoring and observability
│   ├── prometheus/              # Prometheus configuration
│   ├── grafana/                 # Grafana dashboards
│   └── alerting/                # Alerting rules and configs
│
├── notebooks/                   # Jupyter notebooks
│   ├── exploratory/             # Data exploration and experimentation
│   ├── production/              # Production-ready notebooks
│   │   └── nids_autoencoder_production.ipynb
│   └── archive/                 # Archived/old notebooks
│
├── artifacts/                   # Model artifacts and outputs
│   ├── models/                  # Trained model files
│   ├── checkpoints/             # Training checkpoints
│   ├── experiments/             # Experiment artifacts
│   └── exports/                 # Model export formats (ONNX, TensorRT)
│
├── reports/                     # Generated reports and analysis
│   ├── model_performance/       # Model evaluation reports
│   ├── data_quality/            # Data quality reports
│   └── autoencoder_performance_results.csv
│
├── scripts/                     # Utility and automation scripts
│   ├── data_preprocessing.py    # Data preprocessing pipeline
│   ├── model_training.py        # Model training orchestration
│   ├── deployment_utils.py      # Deployment utilities
│   └── maintenance/             # Maintenance and cleanup scripts
│
├── src/                         # Source code (main application)
│   ├── __init__.py
│   ├── api/                     # API layer
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── routes/              # API route definitions
│   │   └── middleware/          # Custom middleware
│   ├── core/                    # Core business logic
│   │   ├── __init__.py
│   │   ├── training.py          # Model training logic
│   │   ├── prediction.py        # Prediction logic
│   │   └── evaluation.py        # Model evaluation
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Data preprocessing
│   │   ├── validation.py        # Data validation
│   │   └── feature_engineering.py
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── autoencoder.py       # Autoencoder implementation
│   │   └── base.py              # Base model classes
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logging.py           # Logging utilities
│       └── metrics.py           # Metrics and evaluation
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── fixtures/                # Test fixtures and data
│   └── conftest.py              # Pytest configuration
│
├── tools/                       # Development tools and utilities
│   ├── data_validation.py       # Data validation tools
│   ├── model_comparison.py      # Model comparison utilities
│   └── performance_profiling.py # Performance profiling tools
│
├── .editorconfig               # Editor configuration
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .pylintrc                   # Pylint configuration
├── CHANGELOG.md                # Version history and changes
├── CONTRIBUTING.md             # Contribution guidelines
├── docker-compose.yml          # Multi-container Docker setup
├── Dockerfile                  # Production Docker image
├── LICENSE                     # Project license
├── Makefile                    # Build automation
├── pyproject.toml              # Python project configuration
├── pytest.ini                 # Pytest configuration
├── README.md                   # Project documentation
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── SECURITY.md                 # Security policy
├── setup.py                    # Python package setup
└── tox.ini                     # Testing across Python versions
```

## 🏗️ Directory Purposes and Standards

### Core Application (`src/`)
- **api/**: REST API implementation using FastAPI
- **core/**: Core business logic and domain models
- **data/**: Data processing and ETL pipelines
- **models/**: ML model definitions and implementations
- **utils/**: Shared utilities and helper functions

### Development & Testing
- **tests/**: Comprehensive test suite with unit, integration, and e2e tests
- **tools/**: Development tools and utility scripts
- **scripts/**: Automation and maintenance scripts

### Documentation & Configuration
- **docs/**: Comprehensive project documentation
- **config/**: Environment-specific configuration files
- **examples/**: Usage examples and tutorials

### Infrastructure & Deployment
- **deployment/**: Deployment configurations and IaC
- **k8s/**: Kubernetes manifests and configurations
- **monitoring/**: Observability and monitoring setup
- **environments/**: Development environment configurations

### Data & Artifacts
- **data/**: Organized data storage by processing stage
- **artifacts/**: Model artifacts, checkpoints, and exports
- **reports/**: Generated reports and analysis results

### CI/CD & Quality
- **.github/**: GitHub Actions workflows and templates
- **.vscode/**: VS Code workspace configuration
- **.devcontainer/**: Development container setup

## 📋 File Naming Conventions

### General Rules
- Use lowercase with underscores for Python files: `data_processor.py`
- Use kebab-case for directories: `model-artifacts/`
- Use descriptive names that clearly indicate purpose
- Include version numbers for configuration files: `config-v1.yaml`

### Specific Conventions
- **Python modules**: `snake_case.py`
- **Configuration files**: `environment.yaml`
- **Docker files**: `Dockerfile.env` for environment-specific
- **Scripts**: `action_description.py` or `action_description.sh`
- **Documentation**: `UPPERCASE.md` for root-level docs
- **Test files**: `test_module_name.py`

## 🔒 Security Considerations

### Sensitive Files
- All sensitive files are properly .gitignored
- Environment variables use `.env.example` as template
- Secrets are managed through proper secret management systems
- API keys and credentials are never committed to version control

### Access Control
- Proper file permissions are maintained
- Development containers run as non-root users
- Production images follow security best practices

## 🚀 Getting Started

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd nids-autoencoder
   make install
   ```

2. **Development Environment**:
   ```bash
   # Using VS Code Dev Container (recommended)
   code .  # Open in VS Code and select "Reopen in Container"
   
   # Or manual setup
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

3. **Quick Start**:
   ```bash
   make help                    # Show all available commands
   make test                    # Run test suite
   make format                  # Format code
   make docker-build           # Build Docker image
   make docker-compose-up       # Start all services
   ```

## 📖 Documentation Structure

- **README.md**: Project overview and quick start
- **docs/**: Comprehensive documentation
- **CONTRIBUTING.md**: Development guidelines
- **SECURITY.md**: Security policies and procedures
- **CHANGELOG.md**: Version history and changes

This structure follows enterprise standards and provides:
- ✅ Clear separation of concerns
- ✅ Scalable architecture
- ✅ Comprehensive testing framework
- ✅ Production-ready deployment
- ✅ Excellent developer experience
- ✅ Security best practices
- ✅ Comprehensive documentation

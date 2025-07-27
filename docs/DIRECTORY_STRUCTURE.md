# NIDS Autoencoder Project - Directory Structure & .gitkeep Files
# ================================================================

## 📁 Project Directory Structure

This document outlines the directory structure maintained by `.gitkeep` files to ensure essential directories are tracked by Git even when empty.

### 🗂️ Directory Structure Overview

```
NIDS-Autoencoder/
├── dataset/                    # Main data directory
│   ├── .gitkeep               # Keeps dataset directory structure
│   ├── raw/                   # Original, unprocessed data
│   │   └── .gitkeep
│   ├── processed/             # Cleaned and preprocessed data
│   │   └── .gitkeep
│   ├── external/              # External datasets and references
│   │   └── .gitkeep
│   └── CIDDS-001-external-week3_1.csv  # Main dataset (tracked)
├── models/                     # Model storage directory
│   ├── .gitkeep               # Keeps models directory structure
│   ├── checkpoints/           # Training checkpoints
│   │   └── .gitkeep
│   ├── experiments/           # Experimental models
│   │   └── .gitkeep
│   └── production_autoencoder.pkl  # Production model (tracked)
├── logs/                      # Logging directory
│   ├── .gitkeep               # Keeps logs directory structure
│   ├── training/              # Training process logs
│   │   └── .gitkeep
│   ├── inference/             # Model inference logs
│   │   └── .gitkeep
│   └── monitoring/            # System monitoring logs
│       └── .gitkeep
└── tests/                     # Testing directory
    ├── .gitkeep               # Keeps tests directory structure
    ├── unit/                  # Unit tests
    │   └── .gitkeep
    ├── integration/           # Integration tests
    │   └── .gitkeep
    └── fixtures/              # Test data and fixtures
        └── .gitkeep
```

### 📋 .gitkeep Files Created

#### 🗃️ Data Management
- **`dataset/.gitkeep`** - Main dataset directory structure
- **`dataset/raw/.gitkeep`** - Raw, unprocessed data storage
- **`dataset/processed/.gitkeep`** - Preprocessed and feature-engineered data
- **`dataset/external/.gitkeep`** - External datasets and benchmarks

#### 🤖 Model Management  
- **`models/.gitkeep`** - Main models directory structure
- **`models/checkpoints/.gitkeep`** - Training checkpoint storage
- **`models/experiments/.gitkeep`** - Experimental model versions

#### 📊 Logging Structure
- **`logs/.gitkeep`** - Main logging directory structure
- **`logs/training/.gitkeep`** - Training process logs
- **`logs/inference/.gitkeep`** - Model inference and prediction logs
- **`logs/monitoring/.gitkeep`** - System and performance monitoring logs

#### 🧪 Testing Structure
- **`tests/.gitkeep`** - Main testing directory structure
- **`tests/unit/.gitkeep`** - Unit tests for individual components
- **`tests/integration/.gitkeep`** - Integration and workflow tests
- **`tests/fixtures/.gitkeep`** - Test data and mock fixtures

### 🎯 Purpose of Each Directory

#### Data Directories (`dataset/`)
| Directory | Purpose | Content Examples |
|-----------|---------|------------------|
| `raw/` | Original datasets | PCAP files, raw CSV, unmodified data |
| `processed/` | Cleaned data | Feature-engineered datasets, normalized data |
| `external/` | Third-party data | Threat intelligence, benchmark datasets |

#### Model Directories (`models/`)
| Directory | Purpose | Content Examples |
|-----------|---------|------------------|
| `checkpoints/` | Training saves | Intermediate model states, recovery points |
| `experiments/` | Research models | A/B test models, architecture experiments |

#### Logging Directories (`logs/`)
| Directory | Purpose | Content Examples |
|-----------|---------|------------------|
| `training/` | Training logs | Loss curves, training metrics, hyperparameter logs |
| `inference/` | Prediction logs | API requests, response times, detection results |
| `monitoring/` | System logs | Health checks, performance metrics, alerts |

#### Testing Directories (`tests/`)
| Directory | Purpose | Content Examples |
|-----------|---------|------------------|
| `unit/` | Component tests | Function tests, class tests, module tests |
| `integration/` | Workflow tests | Pipeline tests, API integration tests |
| `fixtures/` | Test data | Mock responses, sample data, test configurations |

### 🔧 .gitignore Integration

The `.gitignore` file has been updated to:

1. **Exclude content** but **preserve structure**:
   ```gitignore
   # Exclude data files but keep structure
   dataset/*.csv
   !dataset/.gitkeep
   !dataset/*/.gitkeep
   ```

2. **Ensure .gitkeep files are never ignored**:
   ```gitignore
   # Always include structure files
   !**/.gitkeep
   ```

3. **Allow specific important files**:
   ```gitignore
   # Keep main dataset and production model
   !dataset/CIDDS-001-external-week3_1.csv
   !models/production_autoencoder.pkl
   ```

### ✅ Benefits of This Structure

#### 🏗️ **Consistent Project Setup**
- New team members get complete directory structure
- Development environments are immediately ready
- No manual directory creation needed

#### 🔄 **CI/CD Pipeline Ready**
- Build processes can rely on directory existence
- Automated deployments work without manual setup
- Docker containers get proper directory structure

#### 📦 **Clean Repository Management**
- Important directories are preserved
- Large files are excluded but structure remains
- Easy to understand project organization

#### 🚀 **Production Deployment**
- Application expects directories to exist
- Logging and monitoring work immediately
- Model storage directories are ready

### 🎯 Usage Guidelines

#### For Developers:
1. **Never delete .gitkeep files** - They maintain project structure
2. **Add content to directories** - Files will coexist with .gitkeep
3. **Follow naming conventions** - Use established directory purposes

#### For CI/CD:
1. **Directories exist by default** - No need to create them in scripts
2. **Reliable file paths** - Applications can depend on directory structure
3. **Consistent environments** - Same structure across dev/staging/prod

#### For Deployment:
1. **Ready-to-run structure** - Applications start without setup
2. **Proper logging paths** - Log files can be written immediately
3. **Model storage ready** - Model loading/saving works out of the box

### 🔍 Verification Commands

```bash
# Check .gitkeep files are tracked
find . -name ".gitkeep" -type f

# Verify directory structure
tree -a -I '.git|.venv|__pycache__|*.pyc'

# Confirm .gitkeep files in git
git ls-files | grep .gitkeep
```

This structure ensures the NIDS Autoencoder project maintains a professional, production-ready directory organization that supports all development, testing, and deployment workflows.

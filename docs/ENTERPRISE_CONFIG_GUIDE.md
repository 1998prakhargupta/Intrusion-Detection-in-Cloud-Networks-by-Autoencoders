# Enterprise Configuration Management System

## Overview

The NIDS Autoencoder project now includes a comprehensive enterprise-grade configuration management system that centralizes all application settings, supports multiple environments, and provides robust validation.

## 🌟 Key Features

### ✅ Centralized Configuration
- **Single Source of Truth**: All configuration settings managed in one place
- **Hierarchical Structure**: Organized configuration with logical groupings
- **Type Safety**: Strong typing with dataclasses and validation

### 🌍 Environment Management
- **Multi-Environment Support**: Development, Staging, Production, Testing
- **Environment-Specific Optimizations**: Tailored settings per environment
- **Automatic Environment Detection**: Smart environment detection and loading

### 🔧 Configuration Components
- **Model Architecture**: Neural network configuration and hyperparameters
- **Training Settings**: Learning rates, batch sizes, epochs, and optimization
- **Data Management**: Paths, processing options, and validation rules
- **API Configuration**: Server settings, rate limiting, and CORS
- **Database Settings**: Connection strings and pooling configuration
- **Security Settings**: Authentication, encryption, and access control
- **Compute Resources**: GPU/CPU selection and resource allocation
- **Monitoring**: Logging, metrics, and alerting configuration

## 📁 File Structure

```
config/
├── development.yaml       # Development environment settings
├── production.yaml        # Production environment settings  
├── staging.yaml          # Staging environment settings
├── testing.yaml          # Testing environment settings
├── api_config.yaml       # Legacy API configuration
└── train_config.yaml     # Legacy training configuration

src/utils/
├── enterprise_config.py  # Core configuration management
├── config_validation.py  # Pydantic validation schemas
├── config.py            # Legacy configuration (deprecated)
└── simple_config.py     # Simplified configuration utilities

scripts/
└── config_cli.py         # Configuration management CLI tools
```

## 🚀 Quick Start

### Basic Usage

```python
from utils.enterprise_config import ConfigurationManager

# Initialize with automatic environment detection
config_manager = ConfigurationManager()

# Access configuration sections
model_config = config_manager.config.model
training_config = config_manager.config.training
api_config = config_manager.config.api

# Get environment-specific paths
data_path = config_manager.get_data_path("external")
model_path = config_manager.get_model_path("production")
```

### Environment-Specific Loading

```python
from utils.enterprise_config import ConfigurationManager, Environment

# Load specific environment
config_manager = ConfigurationManager(environment=Environment.PRODUCTION)

# Load from custom configuration file
config_manager = ConfigurationManager(config_path="custom_config.yaml")

# Override environment from file
config_manager = ConfigurationManager(
    config_path="production.yaml",
    environment=Environment.STAGING  # Override file's environment
)
```

### Configuration Validation

```python
# Validate configuration
is_valid, errors = config_manager.validate_configuration()
if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")

# Get configuration summary
summary = config_manager.get_config_summary()
print(f"Environment: {summary['environment']}")
print(f"Model: {summary['model']}")
```

## 🛠️ CLI Tools

The configuration system includes powerful CLI tools for management:

### Generate Environment Configurations
```bash
python scripts/config_cli.py generate --output-dir config/environments
```

### Validate Configuration
```bash
python scripts/config_cli.py validate --config config/production.yaml
```

### Convert Configuration Formats
```bash
python scripts/config_cli.py convert --input config.yaml --output config.json
```

### Update Configuration Values
```bash
python scripts/config_cli.py update --config config.yaml \
    --set training.epochs=100 \
    --set api.port=8080
```

### Show Configuration
```bash
python scripts/config_cli.py show --config config.yaml --section training
```

## 🌍 Environment Configurations

### Development Environment
- **Purpose**: Local development and debugging
- **Features**: Debug logging, relaxed validation, development data paths
- **Optimizations**: Fast startup, detailed error messages

### Staging Environment
- **Purpose**: Pre-production testing and validation
- **Features**: Production-like settings with enhanced monitoring
- **Optimizations**: Balanced performance and debugging capabilities

### Production Environment
- **Purpose**: Live deployment and operations
- **Features**: Optimized performance, security hardening, minimal logging
- **Optimizations**: Maximum throughput, resource efficiency

### Testing Environment
- **Purpose**: Automated testing and CI/CD pipelines
- **Features**: Minimal configuration, fast execution, isolated environment
- **Optimizations**: Speed and repeatability

## 📊 Configuration Sections

### Model Configuration
```yaml
model:
  architecture: "autoencoder"
  input_size: 100
  hidden_size: 64
  activation: "relu"
  dropout_rate: 0.1
  batch_norm: true
  regularization:
    type: "l2"
    lambda: 0.001
```

### Training Configuration
```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping:
    enabled: true
    patience: 15
    min_delta: 0.0001
  validation_split: 0.2
```

### Data Configuration
```yaml
data:
  paths:
    external: "/data/external"
    processed: "/data/processed"
    raw: "/data/raw"
  preprocessing:
    normalize: true
    handle_missing: "drop"
    feature_selection: true
  validation:
    max_file_size_mb: 500
    required_columns: ["timestamp", "src_ip", "dst_ip"]
```

### API Configuration
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  rate_limiting:
    enabled: true
    requests_per_minute: 60
  cors:
    enabled: true
    origins: ["http://localhost:3000"]
```

## 🔒 Security Features

### Configuration Validation
- **Schema Validation**: Pydantic schemas ensure type safety
- **Business Rules**: Custom validation for business logic
- **Environment Constraints**: Environment-specific validation rules

### Secure Defaults
- **Production Hardening**: Secure defaults for production environments
- **Access Control**: Configuration-based access restrictions
- **Encryption Support**: Built-in support for encrypted configuration values

### Audit Trail
- **Configuration Changes**: Track all configuration modifications
- **Environment Transitions**: Log environment switches and updates
- **Validation Results**: Record validation successes and failures

## 🔧 Integration with Existing Modules

### Data Loader Integration
```python
from data.loader import DataLoader
from utils.enterprise_config import ConfigurationManager

config_manager = ConfigurationManager()
loader = DataLoader(config_manager=config_manager)

# Uses configured data paths and validation rules
data = loader.load_and_validate_data()
```

### Model Trainer Integration
```python
from core.trainer import ModelTrainer
from utils.enterprise_config import ConfigurationManager

config_manager = ConfigurationManager()
trainer = ModelTrainer(config_manager=config_manager)

# Uses configured model architecture and training parameters
model = trainer.train(train_data, val_data)
```

### API Integration
```python
from api.main import create_app
from utils.enterprise_config import ConfigurationManager

config_manager = ConfigurationManager()
app = create_app(config_manager=config_manager)

# Uses configured API settings and security parameters
```

## 📈 Benefits

### For Development
- **Consistency**: Standardized configuration across all modules
- **Debugging**: Easy configuration inspection and modification
- **Flexibility**: Environment-specific settings without code changes

### For Operations
- **Deployment**: Simplified deployment with environment-aware configuration
- **Monitoring**: Built-in configuration validation and health checks
- **Maintenance**: Centralized configuration management and updates

### For Security
- **Validation**: Comprehensive configuration validation prevents misconfigurations
- **Audit**: Complete audit trail of configuration changes
- **Isolation**: Environment-specific security settings and access controls

## 🎯 Best Practices

### Configuration Management
1. **Use Environment Variables**: Override configuration with environment variables
2. **Validate Early**: Validate configuration at application startup
3. **Document Changes**: Document all configuration modifications
4. **Version Control**: Track configuration changes in version control

### Environment Management
1. **Separate Concerns**: Use different configurations for different environments
2. **Test Configurations**: Validate configurations in testing environment
3. **Secure Production**: Use secure defaults and validation for production
4. **Monitor Changes**: Monitor configuration changes in production

### Development Workflow
1. **Local Development**: Use development environment for local work
2. **Feature Testing**: Test features in staging environment
3. **Production Deployment**: Deploy validated configurations to production
4. **Rollback Support**: Maintain ability to rollback configuration changes

## 🔄 Migration from Legacy Configuration

### Automatic Migration
The enterprise configuration system provides automatic migration from legacy configuration:

```python
# Legacy configuration is automatically detected and migrated
config_manager = ConfigurationManager()

# Access legacy settings through new interface
old_setting = config_manager.get_legacy_setting("model.hidden_size")
```

### Migration Checklist
- [ ] Update module imports to use ConfigurationManager
- [ ] Replace direct config access with config_manager.config
- [ ] Update environment variable references
- [ ] Validate migrated configurations
- [ ] Test in development environment
- [ ] Deploy to staging for validation
- [ ] Deploy to production with monitoring

## 🚀 Next Steps

### Immediate Actions
1. **Integration Testing**: Test configuration system with all modules
2. **Environment Setup**: Configure development, staging, and production environments
3. **CLI Training**: Train team on configuration CLI tools
4. **Documentation**: Update module documentation for configuration usage

### Future Enhancements
1. **Dynamic Configuration**: Support for runtime configuration updates
2. **Configuration Templates**: Provide templates for common configurations
3. **Advanced Validation**: Enhanced validation rules and constraints
4. **Configuration API**: REST API for configuration management
5. **Configuration UI**: Web interface for configuration management

## 📞 Support

For questions about the enterprise configuration system:
- Review this documentation
- Check configuration validation results
- Use CLI tools for configuration inspection
- Consult module integration examples
- Follow best practices for environment management

The enterprise configuration system provides a robust foundation for scalable, maintainable, and secure NIDS deployment across all environments.

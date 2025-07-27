# ===============================================
# NIDS Autoencoder Project - Production Makefile
# ===============================================

# ============ Variables ============
SHELL := /bin/bash
PROJECT_NAME := nids-autoencoder
PYTHON := python3
PIP := pip3
VENV_NAME := .venv
VENV_ACTIVATE := $(VENV_NAME)/bin/activate
DOCKER_IMAGE := nids-autoencoder
DOCKER_TAG := latest
DOCKER_REGISTRY := localhost:5000

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
SCRIPTS_DIR := scripts
MODELS_DIR := models
LOGS_DIR := logs
DATA_DIR := dataset

# Files
REQUIREMENTS := requirements.txt
REQUIREMENTS_DEV := requirements-dev.txt
SETUP_PY := setup.py
DOCKERFILE := Dockerfile
DOCKER_COMPOSE := docker-compose.yml

# ============ Colors for Output ============
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m  # No Color

# ============ Default Target ============
.DEFAULT_GOAL := help

# ============ Help Target ============
.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)NIDS Autoencoder Project - Available Commands$(NC)"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*##/ { printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ============ Environment Setup ============
.PHONY: install
install: venv install-deps ## Create virtual environment and install dependencies
	@echo "$(GREEN)✅ Installation completed successfully!$(NC)"

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(BLUE)🔧 Creating virtual environment...$(NC)"
	@test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	@echo "$(GREEN)✅ Virtual environment created$(NC)"

.PHONY: install-deps
install-deps: venv ## Install project dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@. $(VENV_ACTIVATE) && $(PIP) install --upgrade pip setuptools wheel
	@. $(VENV_ACTIVATE) && $(PIP) install -r $(REQUIREMENTS)
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

.PHONY: install-dev
install-dev: venv ## Install development dependencies
	@echo "$(BLUE)🛠️  Installing development dependencies...$(NC)"
	@. $(VENV_ACTIVATE) && $(PIP) install --upgrade pip setuptools wheel
	@. $(VENV_ACTIVATE) && $(PIP) install -r $(REQUIREMENTS_DEV)
	@. $(VENV_ACTIVATE) && $(PIP) install -e .
	@echo "$(GREEN)✅ Development environment ready$(NC)"

.PHONY: update-deps
update-deps: venv ## Update all dependencies
	@echo "$(BLUE)🔄 Updating dependencies...$(NC)"
	@. $(VENV_ACTIVATE) && $(PIP) install --upgrade pip
	@. $(VENV_ACTIVATE) && $(PIP) install --upgrade -r $(REQUIREMENTS)
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

# ============ Code Quality ============
.PHONY: format
format: venv ## Format code using black and isort
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@. $(VENV_ACTIVATE) && black $(SRC_DIR) $(TEST_DIR)
	@. $(VENV_ACTIVATE) && isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✅ Code formatted$(NC)"

.PHONY: lint
lint: venv ## Run linting checks
	@echo "$(BLUE)🔍 Running linting checks...$(NC)"
	@. $(VENV_ACTIVATE) && flake8 $(SRC_DIR) $(TEST_DIR)
	@. $(VENV_ACTIVATE) && pylint $(SRC_DIR)
	@. $(VENV_ACTIVATE) && mypy $(SRC_DIR)
	@echo "$(GREEN)✅ Linting completed$(NC)"

.PHONY: security
security: venv ## Run security checks
	@echo "$(BLUE)🔒 Running security checks...$(NC)"
	@. $(VENV_ACTIVATE) && bandit -r $(SRC_DIR)
	@. $(VENV_ACTIVATE) && safety check
	@echo "$(GREEN)✅ Security checks completed$(NC)"

.PHONY: check-all
check-all: format lint security ## Run all code quality checks
	@echo "$(GREEN)✅ All checks completed successfully!$(NC)"

# ============ Testing ============
.PHONY: test
test: venv ## Run tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@. $(VENV_ACTIVATE) && pytest $(TEST_DIR) -v

.PHONY: test-cov
test-cov: venv ## Run tests with coverage
	@echo "$(BLUE)🧪 Running tests with coverage...$(NC)"
	@. $(VENV_ACTIVATE) && pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

.PHONY: test-fast
test-fast: venv ## Run fast tests only
	@echo "$(BLUE)⚡ Running fast tests...$(NC)"
	@. $(VENV_ACTIVATE) && pytest $(TEST_DIR) -m "not slow" -v

.PHONY: test-watch
test-watch: venv ## Run tests in watch mode
	@echo "$(BLUE)👀 Running tests in watch mode...$(NC)"
	@. $(VENV_ACTIVATE) && pytest-watch $(TEST_DIR)

# ============ Data & Models ============
.PHONY: download-data
download-data: ## Download sample dataset
	@echo "$(BLUE)📥 Downloading sample data...$(NC)"
	@mkdir -p $(DATA_DIR)
	@bash $(SCRIPTS_DIR)/download_data.sh
	@echo "$(GREEN)✅ Data downloaded$(NC)"

.PHONY: preprocess-data
preprocess-data: venv ## Preprocess training data
	@echo "$(BLUE)🔄 Preprocessing data...$(NC)"
	@. $(VENV_ACTIVATE) && python -m src.data.preprocessor
	@echo "$(GREEN)✅ Data preprocessed$(NC)"

.PHONY: train
train: venv ## Train the autoencoder model
	@echo "$(BLUE)🏋️  Training model...$(NC)"
	@mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	@. $(VENV_ACTIVATE) && python -m src.training.trainer
	@echo "$(GREEN)✅ Model training completed$(NC)"

.PHONY: evaluate
evaluate: venv ## Evaluate model performance
	@echo "$(BLUE)📊 Evaluating model...$(NC)"
	@. $(VENV_ACTIVATE) && python -m src.evaluation.evaluator
	@echo "$(GREEN)✅ Model evaluation completed$(NC)"

# ============ Development ============
.PHONY: notebook
notebook: venv ## Start Jupyter notebook server
	@echo "$(BLUE)📓 Starting Jupyter notebook...$(NC)"
	@. $(VENV_ACTIVATE) && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

.PHONY: clean-notebooks
clean-notebooks: venv ## Clean notebook outputs
	@echo "$(BLUE)🧹 Cleaning notebook outputs...$(NC)"
	@. $(VENV_ACTIVATE) && nbstripout detection_by_Autoencoders.ipynb
	@echo "$(GREEN)✅ Notebooks cleaned$(NC)"

.PHONY: serve
serve: venv ## Start development server
	@echo "$(BLUE)🚀 Starting development server...$(NC)"
	@. $(VENV_ACTIVATE) && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: serve-prod
serve-prod: venv ## Start production server
	@echo "$(BLUE)🚀 Starting production server...$(NC)"
	@. $(VENV_ACTIVATE) && gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# ============ Docker ============
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✅ Docker image built$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	@docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-compose-up
docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)🐳 Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started$(NC)"

.PHONY: docker-compose-down
docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)🐳 Stopping services with docker-compose...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

.PHONY: docker-push
docker-push: docker-build ## Push Docker image to registry
	@echo "$(BLUE)🐳 Pushing Docker image to registry...$(NC)"
	@docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker image pushed$(NC)"

# ============ Documentation ============
.PHONY: docs
docs: venv ## Build documentation
	@echo "$(BLUE)📚 Building documentation...$(NC)"
	@. $(VENV_ACTIVATE) && cd $(DOCS_DIR) && make html
	@echo "$(GREEN)✅ Documentation built$(NC)"

.PHONY: docs-serve
docs-serve: venv ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	@. $(VENV_ACTIVATE) && cd $(DOCS_DIR) && python -m http.server 8080

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	@echo "$(BLUE)🧹 Cleaning documentation...$(NC)"
	@cd $(DOCS_DIR) && make clean
	@echo "$(GREEN)✅ Documentation cleaned$(NC)"

# ============ Monitoring ============
.PHONY: monitor
monitor: ## Start monitoring services
	@echo "$(BLUE)📊 Starting monitoring services...$(NC)"
	@docker-compose -f monitoring/docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✅ Monitoring services started$(NC)"

.PHONY: logs
logs: ## View application logs
	@echo "$(BLUE)📝 Viewing logs...$(NC)"
	@tail -f $(LOGS_DIR)/*.log

.PHONY: prometheus
prometheus: ## Open Prometheus dashboard
	@echo "$(BLUE)📊 Opening Prometheus...$(NC)"
	@open http://localhost:9090 || xdg-open http://localhost:9090

.PHONY: grafana
grafana: ## Open Grafana dashboard
	@echo "$(BLUE)📈 Opening Grafana...$(NC)"
	@open http://localhost:3000 || xdg-open http://localhost:3000

# ============ Deployment ============
.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)🚀 Deploying to staging...$(NC)"
	@bash $(SCRIPTS_DIR)/deploy_staging.sh
	@echo "$(GREEN)✅ Deployed to staging$(NC)"

.PHONY: deploy-prod
deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)🚀 Deploying to production...$(NC)"
	@bash $(SCRIPTS_DIR)/deploy_production.sh
	@echo "$(GREEN)✅ Deployed to production$(NC)"

.PHONY: k8s-deploy
k8s-deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)☸️  Deploying to Kubernetes...$(NC)"
	@kubectl apply -f k8s/
	@echo "$(GREEN)✅ Deployed to Kubernetes$(NC)"

# ============ Cleanup ============
.PHONY: clean
clean: ## Clean build artifacts and caches
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .pytest_cache/ .coverage htmlcov/
	@rm -rf .mypy_cache/ .tox/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cleaned build artifacts$(NC)"

.PHONY: clean-data
clean-data: ## Clean data directories
	@echo "$(BLUE)🧹 Cleaning data directories...$(NC)"
	@rm -rf $(DATA_DIR)/processed/
	@rm -rf $(DATA_DIR)/temp/
	@echo "$(GREEN)✅ Data directories cleaned$(NC)"

.PHONY: clean-models
clean-models: ## Clean model artifacts
	@echo "$(BLUE)🧹 Cleaning model artifacts...$(NC)"
	@rm -rf $(MODELS_DIR)/*.pkl
	@rm -rf $(MODELS_DIR)/checkpoints/
	@echo "$(GREEN)✅ Model artifacts cleaned$(NC)"

.PHONY: clean-logs
clean-logs: ## Clean log files
	@echo "$(BLUE)🧹 Cleaning log files...$(NC)"
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf $(LOGS_DIR)/training/
	@rm -rf $(LOGS_DIR)/inference/
	@echo "$(GREEN)✅ Log files cleaned$(NC)"

.PHONY: clean-all
clean-all: clean clean-data clean-models clean-logs ## Clean everything
	@rm -rf $(VENV_NAME)
	@echo "$(GREEN)✅ Everything cleaned$(NC)"

# ============ Utilities ============
.PHONY: requirements
requirements: venv ## Generate requirements.txt from environment
	@echo "$(BLUE)📋 Generating requirements.txt...$(NC)"
	@. $(VENV_ACTIVATE) && pip freeze > $(REQUIREMENTS)
	@echo "$(GREEN)✅ Requirements generated$(NC)"

.PHONY: check-env
check-env: ## Check environment setup
	@echo "$(BLUE)🔍 Checking environment...$(NC)"
	@echo "Python version: $$(python3 --version)"
	@echo "Pip version: $$(pip3 --version)"
	@echo "Virtual env: $$(test -d $(VENV_NAME) && echo 'Present' || echo 'Missing')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Make: $$(make --version | head -1)"

.PHONY: info
info: ## Show project information
	@echo "$(CYAN)Project Information$(NC)"
	@echo "=================="
	@echo "Name: $(PROJECT_NAME)"
	@echo "Python: $(PYTHON)"
	@echo "Source: $(SRC_DIR)"
	@echo "Tests: $(TEST_DIR)"
	@echo "Docs: $(DOCS_DIR)"
	@echo "Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# ============ CI/CD Helpers ============
.PHONY: ci-setup
ci-setup: install-dev ## Setup CI environment
	@echo "$(BLUE)⚙️  Setting up CI environment...$(NC)"
	@echo "$(GREEN)✅ CI environment ready$(NC)"

.PHONY: ci-test
ci-test: check-all test-cov ## Run CI tests
	@echo "$(GREEN)✅ CI tests completed$(NC)"

.PHONY: pre-commit
pre-commit: format lint test-fast ## Run pre-commit checks
	@echo "$(GREEN)✅ Pre-commit checks passed$(NC)"

# ============ File Dependencies ============
$(VENV_ACTIVATE): $(REQUIREMENTS)
	@make venv
	@make install-deps

# ============ End of Makefile ============

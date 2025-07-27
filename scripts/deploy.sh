#!/bin/bash

# Production deployment script for NIDS Autoencoder System
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="ghcr.io/1998prakhargupta/nids-autoencoder"
DOCKER_TAG="${DOCKER_TAG:-latest}"
NAMESPACE="nids-system"
KUBECTL_TIMEOUT="300s"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" .
    
    if [ $? -eq 0 ]; then
        success "Docker image built successfully: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Push image to registry (if registry is configured)
push_image() {
    if [ -n "${DOCKER_REGISTRY:-}" ]; then
        log "Pushing image to registry..."
        
        docker tag "${DOCKER_IMAGE}:${DOCKER_TAG}" "${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}"
        docker push "${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}"
        
        if [ $? -eq 0 ]; then
            success "Image pushed to registry"
        else
            error "Failed to push image to registry"
            exit 1
        fi
    else
        warning "DOCKER_REGISTRY not set, skipping image push"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log "Deploying to Kubernetes..."
    
    # Create namespace and apply configurations
    kubectl apply -f k8s/namespace-config.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}
    
    # Apply deployment and services
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/ingress-hpa.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available deployment/nids-api -n ${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}
    
    success "Kubernetes deployment completed"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service nids-api-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for testing
    kubectl port-forward service/nids-api-service 8080:80 -n ${NAMESPACE} &
    PF_PID=$!
    
    # Wait a moment for port forward to establish
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        success "Health check passed"
    else
        error "Health check failed"
        kill $PF_PID 2>/dev/null || true
        exit 1
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
}

# Rollback deployment
rollback() {
    warning "Rolling back deployment..."
    kubectl rollout undo deployment/nids-api -n ${NAMESPACE}
    kubectl rollout status deployment/nids-api -n ${NAMESPACE} --timeout=${KUBECTL_TIMEOUT}
    success "Rollback completed"
}

# Main deployment function
deploy() {
    log "Starting NIDS production deployment..."
    
    check_prerequisites
    build_image
    push_image
    deploy_k8s
    health_check
    
    success "Production deployment completed successfully!"
    
    # Display useful information
    echo ""
    log "Deployment Information:"
    echo "  Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    echo "  Namespace: ${NAMESPACE}"
    echo "  Service: nids-api-service"
    echo ""
    log "Useful commands:"
    echo "  kubectl get pods -n ${NAMESPACE}"
    echo "  kubectl logs -f deployment/nids-api -n ${NAMESPACE}"
    echo "  kubectl describe deployment nids-api -n ${NAMESPACE}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "build")
        check_prerequisites
        build_image
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|build|health}"
        echo "  deploy   - Full deployment pipeline (default)"
        echo "  rollback - Rollback to previous version"
        echo "  build    - Build Docker image only"
        echo "  health   - Run health check only"
        exit 1
        ;;
esac

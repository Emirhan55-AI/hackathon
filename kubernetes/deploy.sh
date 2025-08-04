#!/bin/bash

# Aura AI Platform - Kubernetes Deployment Script
# Deploy to Google Kubernetes Engine (GKE) with production configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
CLUSTER_NAME="${CLUSTER_NAME:-aura-ai-cluster}"
REGION="${REGION:-us-central1}"
NODE_COUNT="${NODE_COUNT:-3}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-2}"

echo -e "${BLUE}🚀 Starting Aura AI Platform Kubernetes Deployment${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        print_error "helm not found. Please install Helm."
        exit 1
    fi
    
    print_status "All prerequisites found"
}

# Set up GCP project and enable APIs
setup_gcp() {
    echo -e "${BLUE}🔧 Setting up GCP project...${NC}"
    
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    gcloud services enable container.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable monitoring.googleapis.com
    gcloud services enable logging.googleapis.com
    
    print_status "GCP project configured"
}

# Create GKE cluster
create_cluster() {
    echo -e "${BLUE}🏗️ Creating GKE cluster...${NC}"
    
    # Check if cluster already exists
    if gcloud container clusters describe $CLUSTER_NAME --region=$REGION &> /dev/null; then
        print_warning "Cluster $CLUSTER_NAME already exists. Skipping creation."
        return 0
    fi
    
    # Create the cluster with CPU nodes
    gcloud container clusters create $CLUSTER_NAME \
        --region=$REGION \
        --node-locations=$REGION-a,$REGION-b,$REGION-c \
        --num-nodes=$NODE_COUNT \
        --machine-type=e2-standard-4 \
        --disk-type=pd-ssd \
        --disk-size=50GB \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=10 \
        --enable-autorepair \
        --enable-autoupgrade \
        --enable-ip-alias \
        --network=default \
        --subnetwork=default \
        --cluster-version=latest \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
        --enable-shielded-nodes \
        --enable-network-policy \
        --workload-pool=$PROJECT_ID.svc.id.goog
    
    print_status "GKE cluster created"
}

# Create GPU node pool
create_gpu_nodepool() {
    echo -e "${BLUE}🎮 Creating GPU node pool...${NC}"
    
    # Check if GPU node pool already exists
    if gcloud container node-pools describe gpu-pool --cluster=$CLUSTER_NAME --region=$REGION &> /dev/null; then
        print_warning "GPU node pool already exists. Skipping creation."
        return 0
    fi
    
    gcloud container node-pools create gpu-pool \
        --cluster=$CLUSTER_NAME \
        --region=$REGION \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --num-nodes=$GPU_NODE_COUNT \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=5 \
        --node-taints=nvidia.com/gpu=present:NoSchedule \
        --disk-type=pd-ssd \
        --disk-size=100GB \
        --enable-autorepair \
        --enable-autoupgrade
    
    print_status "GPU node pool created"
}

# Get cluster credentials
get_credentials() {
    echo -e "${BLUE}🔑 Getting cluster credentials...${NC}"
    
    gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
    
    print_status "Cluster credentials configured"
}

# Install NVIDIA GPU drivers
install_gpu_drivers() {
    echo -e "${BLUE}🔧 Installing NVIDIA GPU drivers...${NC}"
    
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    
    print_status "NVIDIA GPU drivers installed"
}

# Install NVIDIA GPU Operator
install_gpu_operator() {
    echo -e "${BLUE}📦 Installing NVIDIA GPU Operator...${NC}"
    
    # Add NVIDIA Helm repository
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install GPU Operator
    helm upgrade --install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator-resources \
        --create-namespace \
        --set driver.enabled=false \
        --wait
    
    print_status "NVIDIA GPU Operator installed"
}

# Install KEDA
install_keda() {
    echo -e "${BLUE}📈 Installing KEDA for autoscaling...${NC}"
    
    # Add KEDA Helm repository
    helm repo add kedacore https://kedacore.github.io/charts
    helm repo update
    
    # Install KEDA
    helm upgrade --install keda kedacore/keda \
        --namespace keda \
        --create-namespace \
        --wait
    
    print_status "KEDA installed"
}

# Install Prometheus Operator
install_prometheus_operator() {
    echo -e "${BLUE}📊 Installing Prometheus Operator...${NC}"
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus Operator
    helm upgrade --install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=fast \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.storageClassName=fast \
        --set grafana.persistence.size=10Gi \
        --wait
    
    print_status "Prometheus Operator installed"
}

# Deploy application manifests
deploy_application() {
    echo -e "${BLUE}🚀 Deploying Aura AI Platform...${NC}"
    
    # Apply manifests in order
    kubectl apply -f namespace.yaml
    kubectl apply -f secrets.yaml
    kubectl apply -f configmaps.yaml
    kubectl apply -f persistent-volumes.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Active namespace/aura-ai --timeout=60s
    
    # Deploy databases first
    kubectl apply -f database-deployments.yaml
    kubectl wait --for=condition=Available deployment/postgresql-deployment -n aura-ai --timeout=300s
    kubectl wait --for=condition=Available deployment/redis-deployment -n aura-ai --timeout=300s
    
    # Deploy Triton Inference Server
    kubectl apply -f triton-deployment.yaml
    kubectl wait --for=condition=Available deployment/triton-inference-server-deployment -n aura-ai --timeout=600s
    
    # Deploy AI services
    kubectl apply -f visual-analysis-deployment.yaml
    kubectl apply -f outfit-recommendation-deployment.yaml
    kubectl apply -f conversational-ai-deployment.yaml
    
    # Wait for AI services to be ready
    kubectl wait --for=condition=Available deployment/visual-analysis-deployment -n aura-ai --timeout=600s
    kubectl wait --for=condition=Available deployment/outfit-recommendation-deployment -n aura-ai --timeout=600s
    kubectl wait --for=condition=Available deployment/conversational-ai-deployment -n aura-ai --timeout=600s
    
    # Deploy gateway
    kubectl apply -f nginx-gateway-deployment.yaml
    kubectl wait --for=condition=Available deployment/nginx-gateway-deployment -n aura-ai --timeout=300s
    
    # Deploy ingress
    kubectl apply -f ingress.yaml
    
    print_status "Application deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    echo -e "${BLUE}📊 Deploying monitoring stack...${NC}"
    
    kubectl apply -f monitoring-stack.yaml
    
    # Wait for monitoring components
    kubectl wait --for=condition=Available deployment/prometheus-deployment -n monitoring --timeout=300s
    kubectl wait --for=condition=Available deployment/grafana-deployment -n monitoring --timeout=300s
    
    print_status "Monitoring stack deployed"
}

# Deploy autoscaling
deploy_autoscaling() {
    echo -e "${BLUE}📈 Deploying KEDA autoscaling...${NC}"
    
    kubectl apply -f keda-autoscaling.yaml
    
    print_status "KEDA autoscaling configured"
}

# Deploy network policies
deploy_security() {
    echo -e "${BLUE}🔒 Deploying network policies...${NC}"
    
    kubectl apply -f network-policies.yaml
    
    print_status "Network policies applied"
}

# Get deployment information
get_deployment_info() {
    echo -e "${BLUE}📋 Deployment Information${NC}"
    
    # Get ingress IP
    INGRESS_IP=$(kubectl get ingress aura-ai-ingress -n aura-ai -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    
    # Get service information
    echo -e "\n${GREEN}📡 Services:${NC}"
    kubectl get services -n aura-ai
    
    echo -e "\n${GREEN}🌐 Ingress:${NC}"
    kubectl get ingress -n aura-ai
    
    echo -e "\n${GREEN}📊 Monitoring:${NC}"
    kubectl get services -n monitoring
    
    echo -e "\n${GREEN}🔗 Access URLs:${NC}"
    if [ "$INGRESS_IP" != "Pending..." ]; then
        echo "API Gateway: https://$INGRESS_IP"
        echo "Visual Analysis: https://$INGRESS_IP/api/v1/visual-analysis/"
        echo "Outfit Recommendation: https://$INGRESS_IP/api/v1/outfit-recommendation/"
        echo "Conversational AI: https://$INGRESS_IP/api/v1/conversational-ai/"
    else
        echo "Ingress IP is still pending. Check back in a few minutes."
    fi
    
    # Get Grafana access info
    GRAFANA_PASSWORD=$(kubectl get secret grafana-admin-secret -n monitoring -o jsonpath='{.data.password}' | base64 -d 2>/dev/null || echo "admin123")
    echo -e "\n${GREEN}📊 Grafana Access:${NC}"
    echo "Username: admin"
    echo "Password: $GRAFANA_PASSWORD"
    echo "URL: Use 'kubectl port-forward -n monitoring svc/grafana-service 3000:3000' then visit http://localhost:3000"
    
    echo -e "\n${GREEN}✅ Deployment completed successfully!${NC}"
}

# Cleanup function
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up resources...${NC}"
    
    read -p "Are you sure you want to delete the entire cluster? This cannot be undone. (y/N): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        gcloud container clusters delete $CLUSTER_NAME --region=$REGION --quiet
        print_status "Cluster deleted"
    else
        echo "Cleanup cancelled"
    fi
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            setup_gcp
            create_cluster
            create_gpu_nodepool
            get_credentials
            install_gpu_drivers
            install_gpu_operator
            install_keda
            deploy_application
            deploy_monitoring
            deploy_autoscaling
            deploy_security
            get_deployment_info
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            get_deployment_info
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|status}"
            echo "  deploy  - Deploy the entire platform"
            echo "  cleanup - Delete the cluster and all resources"
            echo "  status  - Show deployment status and access information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"

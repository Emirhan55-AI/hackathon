# Aura AI Platform - Kubernetes Deployment Guide

## Overview

This directory contains comprehensive Kubernetes manifests for deploying the Aura AI Platform on Google Kubernetes Engine (GKE) with enterprise-grade features including GPU support, autoscaling, monitoring, and security.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GKE Cluster                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   CPU Node Pool │  │  GPU Node Pool  │  │ Monitoring Pool │ │
│  │                 │  │                 │  │                 │ │
│  │ • Nginx Gateway │  │ • Visual Analysis│  │ • Prometheus    │ │
│  │ • Databases     │  │ • Outfit Rec.   │  │ • Grafana       │ │
│  │ • Redis Cache   │  │ • Conv. AI      │  │ • AlertManager  │ │
│  │                 │  │ • Triton Server │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Google Cloud SDK**: Install and configure `gcloud` CLI
2. **kubectl**: Kubernetes command-line tool
3. **Helm**: Package manager for Kubernetes
4. **Docker**: For building custom images (optional)

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Install kubectl
gcloud components install kubectl

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

## Configuration

### Environment Variables

Set these environment variables before deployment:

```bash
export PROJECT_ID="your-gcp-project-id"
export CLUSTER_NAME="aura-ai-cluster"
export REGION="us-central1"
export NODE_COUNT="3"
export GPU_NODE_COUNT="2"
```

### GCP Project Setup

```bash
# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Deployment

### Quick Deployment

Use the automated deployment script:

```bash
chmod +x deploy.sh
./deploy.sh deploy
```

### Manual Deployment

1. **Create GKE Cluster**:
```bash
gcloud container clusters create $CLUSTER_NAME \
    --region=$REGION \
    --num-nodes=$NODE_COUNT \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --enable-network-policy
```

2. **Create GPU Node Pool**:
```bash
gcloud container node-pools create gpu-pool \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --num-nodes=$GPU_NODE_COUNT
```

3. **Get Credentials**:
```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
```

4. **Install Dependencies**:
```bash
# NVIDIA GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm install gpu-operator nvidia/gpu-operator --namespace gpu-operator-resources --create-namespace

# KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace
```

5. **Deploy Application**:
```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f persistent-volumes.yaml
kubectl apply -f database-deployments.yaml
kubectl apply -f triton-deployment.yaml
kubectl apply -f visual-analysis-deployment.yaml
kubectl apply -f outfit-recommendation-deployment.yaml
kubectl apply -f conversational-ai-deployment.yaml
kubectl apply -f nginx-gateway-deployment.yaml
kubectl apply -f ingress.yaml
```

6. **Deploy Monitoring and Autoscaling**:
```bash
kubectl apply -f monitoring-stack.yaml
kubectl apply -f keda-autoscaling.yaml
kubectl apply -f network-policies.yaml
```

## File Descriptions

### Core Infrastructure
- `namespace.yaml`: Kubernetes namespaces for organization
- `secrets.yaml`: Sensitive configuration data
- `configmaps.yaml`: Application configuration
- `persistent-volumes.yaml`: Storage classes and volume claims

### Application Services
- `visual-analysis-deployment.yaml`: DETR-based fashion detection service
- `outfit-recommendation-deployment.yaml`: OutfitTransformer recommendation engine
- `conversational-ai-deployment.yaml`: QLoRA + RAG chat assistant
- `triton-deployment.yaml`: NVIDIA Triton Inference Server
- `nginx-gateway-deployment.yaml`: API Gateway and load balancer
- `database-deployments.yaml`: PostgreSQL and Redis deployments

### Networking and Security
- `ingress.yaml`: External access configuration with SSL
- `network-policies.yaml`: Network segmentation and security rules

### Monitoring and Autoscaling
- `monitoring-stack.yaml`: Prometheus, Grafana, and AlertManager
- `keda-autoscaling.yaml`: Advanced autoscaling based on custom metrics

## Resource Requirements

### Minimum Requirements
- **CPU Nodes**: 3 x e2-standard-4 (4 vCPU, 16GB RAM each)
- **GPU Nodes**: 2 x n1-standard-4 + NVIDIA Tesla T4
- **Storage**: 200GB+ SSD persistent storage
- **Network**: VPC with sufficient IP range

### Production Recommendations
- **CPU Nodes**: 5-10 nodes with autoscaling
- **GPU Nodes**: 3-5 nodes with autoscaling
- **Storage**: 500GB+ with backup strategy
- **Monitoring**: Dedicated monitoring node pool

## Monitoring and Observability

### Prometheus Metrics
Access Prometheus at: `kubectl port-forward -n monitoring svc/prometheus-service 9090:9090`

Key metrics monitored:
- HTTP request rates and latency
- GPU utilization and memory
- Model inference times
- Database connection health
- Cache hit rates

### Grafana Dashboards
Access Grafana at: `kubectl port-forward -n monitoring svc/grafana-service 3000:3000`
- Username: `admin`
- Password: `admin123` (change in production)

Pre-configured dashboards:
- Application Performance Dashboard
- Infrastructure Monitoring Dashboard
- GPU Utilization Dashboard
- Business Metrics Dashboard

### Alerting
AlertManager configuration includes:
- High error rate alerts
- GPU memory warnings
- Slow inference alerts
- Database connectivity issues
- Pod restart notifications

## Autoscaling

### KEDA Scaling Triggers
- **HTTP Requests**: Scale based on request rate
- **GPU Utilization**: Scale when GPU usage is high
- **Queue Depth**: Scale based on pending requests
- **Response Time**: Scale when latency increases
- **WebSocket Connections**: Scale chat service based on active connections

### Horizontal Pod Autoscaler (HPA)
Automatic scaling policies:
- Conservative scale-down (5-10 minutes stabilization)
- Aggressive scale-up (30-60 seconds response)
- Min/max replica limits per service

## Security

### Network Policies
- Default deny-all traffic
- Explicit allow rules for required communication
- Isolation between namespaces
- External access restrictions

### Pod Security
- Non-root containers where possible
- Read-only root filesystems
- Resource limits and requests
- Security contexts with specific UIDs

### Secret Management
- Kubernetes secrets for sensitive data
- Secret rotation capabilities
- No secrets in environment variables
- Encrypted storage at rest

## SSL/TLS Configuration

### Automatic SSL with Google Managed Certificates
```yaml
# Included in ingress.yaml
networking.gke.io/managed-certificates: "aura-ai-ssl-cert"
```

### Custom Domain Setup
1. Point your domain to the ingress IP
2. Update the managed certificate configuration
3. Update ingress hosts configuration

## Troubleshooting

### Common Issues

1. **GPU Nodes Not Ready**:
```bash
kubectl describe nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-t4
kubectl logs -n gpu-operator-resources -l app=nvidia-driver-daemonset
```

2. **Pods Stuck in Pending**:
```bash
kubectl describe pod <pod-name> -n aura-ai
kubectl get events -n aura-ai --sort-by='.lastTimestamp'
```

3. **Autoscaling Not Working**:
```bash
kubectl get hpa -n aura-ai
kubectl describe scaledobject -n aura-ai
kubectl logs -n keda -l app=keda-operator
```

4. **Ingress Not Getting IP**:
```bash
kubectl describe ingress aura-ai-ingress -n aura-ai
kubectl get managedcertificate -n aura-ai
```

### Debugging Commands

```bash
# Check cluster status
kubectl cluster-info

# Check node status
kubectl get nodes -o wide

# Check pod status
kubectl get pods -n aura-ai -o wide

# Check service status
kubectl get services -n aura-ai

# Check ingress status
kubectl get ingress -n aura-ai

# Check persistent volumes
kubectl get pv,pvc -n aura-ai

# Check logs
kubectl logs -f deployment/visual-analysis-deployment -n aura-ai

# Port forward for local testing
kubectl port-forward -n aura-ai svc/nginx-gateway-service 8080:80
```

## Performance Tuning

### GPU Optimization
- TensorRT optimization for inference models
- Batch processing for better GPU utilization
- Model parallelism for large models
- Memory-efficient loading strategies

### Database Tuning
- Connection pooling configuration
- Index optimization for query performance
- Read replicas for scaling reads
- Backup and restore strategies

### Cache Optimization
- Redis cluster mode for high availability
- Cache warming strategies
- TTL optimization for different data types
- Memory usage monitoring

## Backup and Disaster Recovery

### Database Backup
```bash
# Automated backup job included in database-deployments.yaml
kubectl get cronjobs -n aura-ai
```

### Model Backup
- Models stored in Google Cloud Storage
- Automatic versioning and rollback
- Cross-region replication

### Configuration Backup
```bash
# Backup all configurations
kubectl get all,configmap,secret -n aura-ai -o yaml > backup.yaml
```

## Cost Optimization

### Node Pool Management
- Use preemptible instances for non-critical workloads
- Implement cluster autoscaling
- Schedule workloads during off-peak hours
- Right-size instances based on actual usage

### Storage Optimization
- Use appropriate storage classes
- Implement data lifecycle policies
- Regular cleanup of unused volumes
- Compression for backup data

## Production Checklist

- [ ] SSL certificates configured
- [ ] Domain name pointing to ingress
- [ ] Monitoring alerts configured
- [ ] Backup strategy implemented
- [ ] Security policies applied
- [ ] Resource limits set
- [ ] Autoscaling tested
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team training completed

## Support and Maintenance

### Regular Maintenance Tasks
1. Update Kubernetes cluster version
2. Update container images
3. Review and rotate secrets
4. Check resource utilization
5. Review monitoring alerts
6. Update documentation

### Monitoring Checklist
- [ ] All pods are running
- [ ] Ingress has external IP
- [ ] SSL certificates are valid
- [ ] Monitoring stack is healthy
- [ ] Autoscaling is working
- [ ] Database connections are healthy
- [ ] Cache is responding
- [ ] GPU utilization is optimal

For additional support, refer to the main project documentation or contact the development team.

# Compute Environments Setup for RecSys Challenge 2024

## Google Cloud Platform Configuration

### Prerequisites
```bash
# Enable required services
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Create storage bucket
gsutil mb -l us-central1 gs://recsys-2024-artifacts
```

## Virtual Machine Specifications

### 1. KAMI Pipeline (CPU + High Memory)
**Machine Type**: `n2-highmem-96`
- **OS**: Debian GNU/Linux 11
- **CPU**: Intel Xeon @ 2.80GHz, 96 vCPUs  
- **RAM**: 768 GB
- **Storage**: 500GB SSD
- **GPU**: None

**Creation Command**:
```bash
gcloud compute instances create kami-vm \
  --machine-type=n2-highmem-96 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd \
  --zone=us-central1-a
```

**Setup Commands**:
```bash
gcloud compute ssh kami-vm
sudo apt update
sudo apt install -y git docker.io docker-compose python3-pip
sudo usermod -aG docker $USER
# Logout and log back in for Docker permissions
```

### 2. KFujikawa Pipeline (GPU + Neural Networks)
**Machine Type**: `g2-standard-32`
- **OS**: Ubuntu 20.04.2
- **CPU**: 32 vCPUs
- **GPU**: NVIDIA L4 x 1
- **RAM**: 128 GB
- **Storage**: 300GB SSD
- **CUDA**: 12.5

**Creation Command**:
```bash
gcloud compute instances create kfujikawa-vm \
  --machine-type=g2-standard-32 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --zone=us-central1-a
```

**Setup Commands**:
```bash
gcloud compute ssh kfujikawa-vm
sudo apt update
sudo apt install -y git python3-pip

# Install NVIDIA drivers and CUDA 12.5
sudo apt install -y nvidia-driver-555
sudo reboot
# After reboot, install CUDA toolkit
```

### 3. Sugawarya Pipeline (CPU + Extreme Memory)
**Machine Type**: `c2d-highmem-112`
- **OS**: Debian GNU/Linux 11
- **CPU**: AMD EPYC 7B13, 112 vCPUs
- **RAM**: 896 GB
- **Storage**: 500GB SSD
- **Python**: 3.12.1

**Creation Command**:
```bash
gcloud compute instances create sugawarya-vm \
  --machine-type=c2d-highmem-112 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd \
  --zone=us-central1-a
```

**Setup Commands**:
```bash
gcloud compute ssh sugawarya-vm
sudo apt update
sudo apt install -y git python3.12 python3-pip
pip install -U poetry
```

## Cost Management

### Important Commands
```bash
# Start instances
gcloud compute instances start INSTANCE_NAME --zone=us-central1-a

# Stop instances (CRITICAL for cost control)
gcloud compute instances stop INSTANCE_NAME --zone=us-central1-a

# Delete instances when done
gcloud compute instances delete INSTANCE_NAME --zone=us-central1-a
```

### Estimated Costs (per hour when running)
- **KAMI (n2-highmem-96)**: ~$4.60/hour
- **KFujikawa (g2-standard-32)**: ~$3.20/hour  
- **Sugawarya (c2d-highmem-112)**: ~$5.40/hour

**⚠️ Always stop instances immediately after use to avoid charges!**
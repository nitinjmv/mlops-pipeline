# MLOps Pipeline with GitHub Actions, Minikube, and Kubernetes

This repository implements a basic MLOps pipeline using CI/CD workflows to automate the build and deployment of a machine learning application. The deployment targets a Minikube cluster running on a Google Cloud VM.

---

## 📦 Repository Structure

```
.
├── .github/
│   ├── workflows/
│   │   ├── ci-pipeline.yml       # CI workflow: Build, test, Dockerize, push image
│   │   └── cd-pipeline.yml       # CD workflow: SSH into VM, apply Kubernetes resources
├── k8s/
│   ├── deployment.yaml           # Kubernetes Deployment resource
│   └── service.yaml              # Kubernetes Service (NodePort)
└── README.md
```

---

## 🚀 Workflow Overview

### 1. Code Commit (Trigger)

- Developer pushes code to the main branch.
- GitHub Actions triggers the CI pipeline (`ci-pipeline.yml`).

### 2. Continuous Integration (CI)

- Runs tests and linting (if configured).
- Builds a Docker image of the ML app.
- Pushes the image to a container registry (Docker Hub, GCR, etc.).

### 3. Continuous Deployment (CD)

- Triggered after a successful CI run.
- SSHs into a GCP VM.
- Deploys the application to Minikube using:
  - `deployment.yaml`
  - `service.yaml`
- The app is exposed via a NodePort service.

---

## 🛠️ Prerequisites

- A Google Cloud VM with:
  - Minikube installed and running
  - Kubernetes configured (`kubectl`)
- Public SSH key added to the VM
- Private SSH key stored as a GitHub secret: `SSH_PRIVATE_KEY`
- Docker image registry credentials stored as secrets if needed

---

## 🔑 Secrets (GitHub)

Add the following repository secrets:

- `SSH_PRIVATE_KEY`: Private key for SSH access to the GCP VM
- `REGISTRY_USERNAME` and `REGISTRY_PASSWORD` (if using a private container registry)

---

## 🌐 Accessing the App

Once deployed, the app can be accessed at:

```
http://<GCP_VM_EXTERNAL_IP>:<NodePort>
```

The default NodePort is defined in `service.yaml` (e.g., `30080`).

---

## 📸 Architecture Diagram

(See attached image in the repository or refer to the generated diagram)

---

## 🤖 Future Enhancements

- Add monitoring (Prometheus/Grafana)
- Use Helm for templated deployments
- Automate Minikube provisioning with Terraform or Ansible
- Enable rolling updates and canary deployments

---

## 📄 License

MIT License

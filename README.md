# Flask ML App - CI/CD with Controlled Canary Deployment

##  Overview

## Controlled Rollout with Canary Strategy for a Deep Learning API

This project showcases a **controlled rollout strategy** for deploying a PyTorch deep learning model via a Flask API. Using GitHub Actions and Docker, it performs **model validation (ONNX)**, **load testing (Locust)**, and a **custom NGINX-based canary deployment** that gradually shifts traffic from a stable release to a new one. 

The system includes built-in **error monitoring and automatic rollback** to ensure safe delivery of ML updates. Ideal for production-grade inference workflows where reliability and resilience are critical.


---


## ⚙️ Architecture Summary

```plaintext
+---------+        +------------------+        +------------------+
| GitHub  | -----> | DockerHub Repo   | -----> | GitHub Actions   |
+---------+        +------------------+        +------------------+
     |
     v
+-------------------+-------------------+--------------------------+
|    Build & Test   |  Validate Model   |     Canary Deployment    |
|  (Docker + Locust)| (ONNX Runtime)    | (NGINX Traffic Splitting)|
+-------------------+-------------------+--------------------------+
                                      |
                                      v
                            +---------------------+
                            |  Tag Stable Version |
                            +---------------------+

```
## 📂 Directory Structure

```plaintext
.
├── app/                 # Flask application source code
├── model/               # Trained ML models (ONNX + PyTorch format)
├── nn_model.py          # PyTorch model class definition
├── validate_onnx.py     # Script for ONNX model validation
├── locustfile.py        # Load testing script using Locust
├── Dockerfile           # Image definition for the Flask app
├── nginx.conf           # NGINX config for traffic splitting
├── .github/workflows/   # GitHub Actions CI/CD workflows
```

---

##  What Each Component Does

### CI Job (`ci`)
1. **Build Docker Image**: Creates the image with the latest Flask app and ML model.
2. **Push to DockerHub**: Publishes image with a unique `canary-deployment-${{ github.sha }}` tag.
3. **Run Flask Locally**: Starts the app in a container to test if it launches correctly.
4. **Health Check**: Ensures `/health` endpoint responds (basic readiness test).
5. **Load Test with Locust**: Simulates requests to assess performance and response integrity.

---

### (Optional) Model Health Check (`model-health-check`)
> *You commented this out during testing, but it’s still valuable.*

1. **Pull Canary Image**: Downloads the newly built Docker image.
2. **Validate ONNX Model**: Uses `onnxruntime` to verify that:
   - The model loads properly.
   - It accepts correct input shape (e.g., `[None, 2]`).
   - Inference works and output is reasonable.

💡 Prevents "it-deployed-but-doesn't-work" problems.

---

### Tag & Push as Stable (`tag-and-push-stable`)
1. **Pull the Canary Image**: Uses the same SHA-tagged image.
2. **Tag as `stable`**: If it passed health checks, tag it for production use.
3. **Push Stable Image**: Publishes the tag `:stable` to DockerHub.

---

### Canary Deployment (`canary-deployment`)
1. **Pull Both Images**: Gets both `stable` and new canary image.
2. **Run Two Containers**:
   - Port `7011`: Old (Stable)
   - Port `7012`: New (Canary)
3. **NGINX Load Balancer**:
   - Reads from `nginx.conf`
   - Initially routes most traffic to `stable`
4. **Gradual Traffic Shift**:
   - Script updates `nginx.conf` to route 10%, 25%, ..., up to 100% to canary
   - After each change, restarts NGINX and waits for observation
5. **Auto Rollback**:
   - Calls `/health` endpoint
   - Checks if `.error_rate` exceeds threshold
   - If errors > 5, rolls back traffic to `stable`

---

## Why ONNX?

Although PyTorch models can be deployed directly, ONNX provides:
- **Cross-framework compatibility**
- **Lightweight, faster inference in production**
- **Language-agnostic** (can be served from C++, Java, Node.js, etc.)
- Easier to test via `onnxruntime` for CI-level checks

---

## Canary Deployment in Action

Your `nginx.conf` file defines a reverse proxy with weights:

```nginx
upstream flask_servers {
    server flask_old:7000 weight=90;
    server flask_canary:7000 weight=10;
}
## Environment Variables

| Variable     | Description                                     |
|--------------|-------------------------------------------------|
| `IMAGE_NAME` | Name of the Docker image (e.g., `flask-app`)    |
| `IMAGE_TAG`  | Unique tag for the canary image (e.g., Git SHA) |
| `REGISTRY`   | DockerHub registry path (typically `docker.io`) |

---

## How to Trigger the Workflow

To trigger the CI/CD pipeline and initiate a canary deployment:

```bash
git checkout canary-deployment
git push origin canary-deployment

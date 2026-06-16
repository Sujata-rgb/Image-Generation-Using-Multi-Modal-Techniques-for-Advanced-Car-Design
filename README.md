# 🚗 Multimodal Car Designing System
### AI-Based Image Retrieval & Generation

A full-stack AI system built with Django that performs **text-to-image retrieval** using CLIP embeddings and **car design generation** using a local Stable Diffusion Img2Img pipeline.

---

## ⚙️ Installation Guide for Windows

We highly recommend using a Virtual Environment (`venv`) so that these packages do not conflict with your system's global Python installation.

### 1. Create a Virtual Environment
Open PowerShell or Command Prompt in the project folder and run:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
```bash
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows Command Prompt (CMD):
.\venv\Scripts\activate.bat
```
*(You should now see `(venv)` at the start of your terminal line).*

### 3. Install Dependencies
This project uses PyTorch with CUDA support (for NVIDIA GPUs) and Hugging Face Diffusers for generation. Install everything by running:
```bash
pip install -r requirements.txt
```
*(Note: PyTorch is a 2.4GB download, so this step may take a few minutes depending on your internet speed).*

### 4. Database Setup (Migrations)
The system uses a secure local SQLite database for user authentication. Initialise it by running:
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create an Admin Account (Optional but recommended)
To access the Django Admin panel and manage users, create a superuser:
```bash
python manage.py createsuperuser
```
Follow the prompts to enter a username, email, and password.

---

## 🚀 Running the System

Start the Django server by running:
```bash
python manage.py runserver
```

Then open your browser at:
```
http://127.0.0.1:8000
```

> **First Run Note:** The very first time you search for an image, it will download the CLIP model (~600 MB). The very first time you generate an image, it will download the Stable Diffusion v1.5 model (~4 GB). Subsequent runs will be instantly loaded from your local cache!

---

## 🎨 Workflow

1. **Landing Page:** Users must log in or sign up to access the application.
2. **Retrieval (Step 1):** Enter a prompt (e.g., "A sleek red sports car"). CLIP retrieves the Top-5 visually matching vehicles from the local dataset.
3. **Generation (Step 2):** Enter a new prompt (e.g., "Make it a matte black cyberpunk vehicle"). Stable Diffusion takes the retrieved image and synthesises a completely new, photorealistic 8k image based on your text prompt!

---

## 🔧 Tech Stack
- **Web Framework:** Django
- **Deep Learning:** PyTorch (CUDA 12.1 enabled)
- **Vision-Language Model:** OpenAI CLIP
- **Generative AI:** Stable Diffusion v1.5 (Hugging Face Diffusers)
- **Frontend:** HTML5 + Vanilla JS (Glassmorphism UI)

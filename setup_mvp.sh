#!/bin/bash
# =========================================================
# Google-Native MVP Environment Setup Script
# =========================================================

# ---- CONFIG ----
PYTHON_VERSION="3.12"    # Change if you want 3.9 or 3.11
VENV_NAME="venv"
GCP_PROJECT_ID="my-ai-project-68045"  # <-- CHANGE THIS

# ---- 1. CHECK PYTHON ----
echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null
then
    echo "Python 3 not found. Please install Python $PYTHON_VERSION."
    exit 1
fi

# ---- 2. CREATE VIRTUAL ENV ----
echo "[2/7] Creating virtual environment..."
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# ---- 3. INSTALL PYTHON DEPENDENCIES ----
echo "[3/7] Installing Python packages..."
pip install --upgrade pip
pip install streamlit pandas chromadb google-cloud-aiplatform python-dotenv google-adk

# --- Step 4: Find Python 3.12 ---
PYTHON_CMD=$(which python3.12)

if [ -z "$PYTHON_CMD" ]; then
    echo "Python 3.12 not found. Please install it and ensure it's in your PATH."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# --- Step 5: Create virtual environment ---
$PYTHON_CMD -m venv venv

# --- Step 6: Activate venv ---
source venv/bin/activate

# --- Step 7 Upgrade pip ---
$PYTHON_CMD -m pip install --upgrade pip

# --- Step 8: Install required packages ---
if [ -f requirements.txt ]; then
    $PYTHON_CMD -m pip install -r requirements.txt
else
    echo "requirements.txt not found, installing default packages..."
    $PYTHON_CMD -m pip install streamlit google-cloud-aiplatform pandas
fi

# ---- 9. INSTALL GCLOUD SDK ----
echo "[4/7] Checking gcloud CLI..."
if ! command -v gcloud &> /dev/null
then
    echo "Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi

# ---- 10. AUTHENTICATE GCLOUD ----
echo "[5/7] Authenticating with Google Cloud..."
gcloud auth login
gcloud auth application-default login

# ---- 11. SET PROJECT & ENABLE APIS ----
echo "[6/7] Setting project and enabling APIs..."
gcloud config set project $GCP_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
gcloud services enable discoveryengine.googleapis.com

# ---- 12. DONE ----
echo "[7/7] Setup complete!"
echo "Activate your environment with:  source $VENV_NAME/bin/activate"
echo "You can now run:  streamlit run app.py"

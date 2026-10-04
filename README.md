# Service Recommendation System - TensorFlow

This academic project demonstrates a small AI-based recommendation system for service businesses. It uses neural collaborative filtering implemented with TensorFlow/Keras. Existing customers receive personalized recommendations from learned user/service embeddings. New customers are handled with a cold-start fallback based on average ratings and interaction frequency.

## Features

- TensorFlow/Keras neural collaborative filtering
- User and service embeddings
- 80/20 train/test split
- Test MSE and MAE reporting
- Personalized Top-N recommendations
- Optional category filter
- Cold-start fallback for unknown customers
- Saved `.keras` model and in-memory model cache
- Synthetic demonstration data only

## Project structure

- `app.py` - command-line interface
- `model.py` - model, evaluation, persistence, and recommendation logic
- `data/services.csv` - service catalog
- `data/interactions.csv` - synthetic ratings
- `requirements.txt` - Python dependencies
- `PROJECT_NOTES.md` - design and learning notes
- `.gitignore` - files excluded from version control

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Setup (macOS/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Testing

Use an existing customer such as `U001` for personalized recommendations. Use an unknown ID such as `U100` to demonstrate cold-start handling. Press Enter at the category prompt to search all categories and press Enter at the number prompt to use five recommendations.

On the first run, the program trains, evaluates, and saves `service_recommender.keras`. Later runs load the saved model instead of retraining it. Within the same process, the model is also reused from memory.

## Academic note

The included data are synthetic and are intended only for demonstration. A production system should use consented customer data, stronger validation, privacy controls, bias monitoring, security, and ongoing performance monitoring.

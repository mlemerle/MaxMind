# Deploy MaxMind to GitHub & Streamlit Cloud

## Prerequisites
- Install Git from https://git-scm.com/download/win
- GitHub account: mlemerle
- Streamlit Cloud account: mlemerle

## Step 1: Initialize Git Repository

```powershell
cd "c:\Users\maxle\OneDrive\Projects\MaxMindTraining"
git init
git add .
git commit -m "Initial commit: MaxMind Cognitive Training Platform"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/mlemerle
2. Click "New repository"
3. Name: `maxmind-training`
4. Description: "Advanced cognitive training platform with iPhone-inspired design"
5. Keep it public for Streamlit Cloud
6. Don't initialize with README (we have one)
7. Click "Create repository"

## Step 3: Connect Local to GitHub

```powershell
git remote add origin https://github.com/mlemerle/maxmind-training.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your mlemerle account
3. Click "New app"
4. Repository: `mlemerle/maxmind-training`
5. Branch: `main`
6. Main file path: `MaxMind.py`
7. App URL: `maxmind-trainer` (or choose your preferred name)
8. Click "Deploy!"

## Step 5: Data Persistence Setup

The app now includes a `storage.py` module that handles persistent data storage:
- Creates local `.maxmind_data` folder for user progress
- Falls back to session state if file storage unavailable
- Maintains user progress across app restarts

## Step 6: iPhone Home Screen App

Users can add your app to iPhone home screen:
1. Open in Safari: https://maxmind-trainer.streamlit.app
2. Tap Share button
3. Select "Add to Home Screen"
4. Acts like a native app!

## Automatic Updates

Any changes you push to GitHub will automatically update the Streamlit app within minutes.

# MaxMind Cognitive Training

An advanced cognitive training platform with spaced repetition, dual N-Back, strategy training, and World Model learning.

## Features

- **Spaced Repetition**: SM-2 algorithm with custom card management
- **Cognitive Drills**: Dual N-Back, Task Switching, Complex Span, Go/No-Go, Processing Speed
- **Learning Modules**: AI topic suggestions, World Model integration
- **Progress Tracking**: 60-day calendar, adaptive difficulty, completion indicators
- **Premium UI**: Three themes (Light/Dark/Blackout) with iPhone-inspired design

## Live Demo

Visit the live app: [MaxMind Trainer](https://maxmind-trainer.streamlit.app)

## iPhone App

Add to your iPhone home screen:
1. Open in Safari
2. Tap Share button
3. Select "Add to Home Screen"
4. Enjoy native app experience!

## Installation

1. Install Python 3.8+ 
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run MaxMind.py
```

The app will open in your browser at `http://localhost:8501`

## Navigation

Use the sidebar to navigate between different modules:
- **Dashboard**: Overview and daily progress
- **Spaced Review**: Flashcard review session
- **Cognitive Drills**: N-Back, Stroop, Complex Span, Go/No-Go
- **Mental Math**: Timed arithmetic practice
- **Writing**: 12-minute writing sprints
- **Forecasts**: Personal prediction tracking
- **Argument Map**: Visual argument structure
- **Settings**: Configuration and data backup

## Data Persistence

All data is stored in browser session state. Use the Settings page to:
- Export your data as JSON backup
- Import previously exported data
- Adjust daily card limits

## Notes

- The app uses browser session storage - your data persists across page refreshes but not browser restarts unless exported
- Graphviz must be installed on your system for argument mapping to work
- Some cognitive drill timings are simulated due to Streamlit's execution model

import tkinter as tk
from tkinter import messagebox, ttk
import random
import threading
import time
from collections import deque
import pyttsx3

class DualNBackTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dual N-Back Test")
        self.root.geometry("700x800")
        self.root.configure(bg='#2c3e50')
        
        # Make window resizable and bring to front
        self.root.resizable(True, True)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        
        # Game parameters
        self.n_level = 2  # Starting n-back level
        self.grid_size = 3
        self.trial_duration = 3.0  # seconds per trial
        self.inter_trial_interval = 0.5  # seconds between trials
        self.num_trials = 20  # trials per round
        
        # Game state
        self.current_trial = 0
        self.is_running = False
        self.is_paused = False
        
        # Stimulus history
        self.position_history = deque(maxlen=10)
        self.audio_history = deque(maxlen=10)
        
        # User responses
        self.position_responses = []
        self.audio_responses = []
        
        # Current stimuli
        self.current_position = None
        self.current_audio = None
        
        # Letters for audio stimuli
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # Statistics
        self.position_correct = 0
        self.audio_correct = 0
        self.position_total = 0
        self.audio_total = 0
        
        # UI setup
        print("Setting up UI...")
        self.setup_ui()
        print("UI setup complete. Window should be visible.")
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Dual N-Back Test", 
                              font=('Arial', 24, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # N-Back level display
        self.level_frame = tk.Frame(self.root, bg='#2c3e50')
        self.level_frame.pack(pady=10)
        
        tk.Label(self.level_frame, text="N-Back Level:", 
                font=('Arial', 14), bg='#2c3e50', fg='white').pack(side=tk.LEFT)
        
        self.level_var = tk.StringVar(value=str(self.n_level))
        level_spinbox = tk.Spinbox(self.level_frame, from_=1, to=9, 
                                  textvariable=self.level_var, width=5,
                                  font=('Arial', 12))
        level_spinbox.pack(side=tk.LEFT, padx=10)
        level_spinbox.config(command=self.update_n_level)
        
        # Instructions
        instructions = """Instructions:
• Watch the grid for position matches and listen for audio matches
• Press SPACE when the position matches the one from N trials back
• Press L when the letter matches the one from N trials back
• You can press both keys if both match
• Try to score >80% on both to advance to the next level"""
        
        instruction_label = tk.Label(self.root, text=instructions, 
                                   font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1',
                                   justify=tk.LEFT)
        instruction_label.pack(pady=10)
        
        # 3x3 Grid
        self.grid_frame = tk.Frame(self.root, bg='#2c3e50')
        self.grid_frame.pack(pady=20)
        
        self.grid_buttons = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                btn = tk.Button(self.grid_frame, text="", width=8, height=4,
                               font=('Arial', 16, 'bold'),
                               bg='#34495e', fg='white',
                               relief='raised', borderwidth=2)
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.grid_buttons.append(row)
        
        # Progress and trial info
        self.info_frame = tk.Frame(self.root, bg='#2c3e50')
        self.info_frame.pack(pady=10)
        
        self.trial_label = tk.Label(self.info_frame, text="Trial: 0/0", 
                                   font=('Arial', 12), bg='#2c3e50', fg='white')
        self.trial_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.info_frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Scores
        self.score_frame = tk.Frame(self.root, bg='#2c3e50')
        self.score_frame.pack(pady=10)
        
        self.position_score_label = tk.Label(self.score_frame, text="Position: 0/0 (0%)", 
                                           font=('Arial', 12), bg='#2c3e50', fg='#3498db')
        self.position_score_label.pack()
        
        self.audio_score_label = tk.Label(self.score_frame, text="Audio: 0/0 (0%)", 
                                        font=('Arial', 12), bg='#2c3e50', fg='#e74c3c')
        self.audio_score_label.pack()
        
        # Control buttons
        print("Creating control buttons...")
        self.button_frame = tk.Frame(self.root, bg='#2c3e50')
        self.button_frame.pack(pady=20)
        
        self.start_button = tk.Button(self.button_frame, text="Start Test", 
                                     command=self.start_test,
                                     font=('Arial', 12, 'bold'),
                                     bg='#27ae60', fg='white', width=12)
        self.start_button.pack(side=tk.LEFT, padx=5)
        print("Start button created")
        
        self.pause_button = tk.Button(self.button_frame, text="Pause", 
                                     command=self.toggle_pause,
                                     font=('Arial', 12, 'bold'),
                                     bg='#f39c12', fg='white', width=12)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        print("Pause button created")
        
        self.stop_button = tk.Button(self.button_frame, text="Stop", 
                                    command=self.stop_test,
                                    font=('Arial', 12, 'bold'),
                                    bg='#e74c3c', fg='white', width=12)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        print("Stop button created")
        print("All buttons should now be visible")
        
        # Key bindings
        self.root.bind('<KeyPress-space>', self.position_match_response)
        self.root.bind('<KeyPress-l>', self.audio_match_response)
        self.root.bind('<KeyPress-L>', self.audio_match_response)
        self.root.focus_set()
        
    def update_n_level(self):
        try:
            self.n_level = int(self.level_var.get())
        except ValueError:
            self.n_level = 2
            self.level_var.set("2")
    
    def start_test(self):
        if self.is_running:
            return
            
        self.update_n_level()
        self.reset_game()
        self.is_running = True
        self.is_paused = False
        
        # Start the test in a separate thread
        self.test_thread = threading.Thread(target=self.run_test)
        self.test_thread.daemon = True
        self.test_thread.start()
    
    def toggle_pause(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")
    
    def stop_test(self):
        self.is_running = False
        self.is_paused = False
        self.pause_button.config(text="Pause")
        self.clear_grid()
        
    def reset_game(self):
        self.current_trial = 0
        self.position_history.clear()
        self.audio_history.clear()
        self.position_responses = []
        self.audio_responses = []
        self.position_correct = 0
        self.audio_correct = 0
        self.position_total = 0
        self.audio_total = 0
        self.clear_grid()
        self.update_display()
    
    def clear_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_buttons[i][j].config(bg='#34495e')
    
    def highlight_position(self, position):
        self.clear_grid()
        if position is not None:
            row, col = position
            self.grid_buttons[row][col].config(bg='#3498db')
    
    def play_audio(self, letter):
        # Use text-to-speech to speak the letter
        def speak_letter():
            self.tts_engine.say(letter)
            self.tts_engine.runAndWait()
        
        # Run TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak_letter)
        tts_thread.daemon = True
        tts_thread.start()
    
    def position_match_response(self, event=None):
        if not self.is_running or self.is_paused:
            return
        
        # Record the response for the current trial
        if self.current_trial < len(self.position_responses):
            return  # Already responded
        
        self.position_responses.append(True)
        
    def audio_match_response(self, event=None):
        if not self.is_running or self.is_paused:
            return
        
        # Record the response for the current trial
        if self.current_trial < len(self.audio_responses):
            return  # Already responded
        
        self.audio_responses.append(True)
    
    def run_test(self):
        for trial in range(self.num_trials):
            if not self.is_running:
                break
                
            # Wait if paused
            while self.is_paused and self.is_running:
                time.sleep(0.1)
                
            if not self.is_running:
                break
            
            self.current_trial = trial + 1
            
            # Generate stimuli
            position = (random.randint(0, self.grid_size-1), 
                       random.randint(0, self.grid_size-1))
            letter = random.choice(self.letters)
            
            self.current_position = position
            self.current_audio = letter
            
            # Add to history
            self.position_history.append(position)
            self.audio_history.append(letter)
            
            # Initialize response lists if needed
            while len(self.position_responses) < trial + 1:
                self.position_responses.append(False)
            while len(self.audio_responses) < trial + 1:
                self.audio_responses.append(False)
            
            # Present stimuli
            self.root.after(0, self.highlight_position, position)
            self.root.after(0, self.play_audio, letter)
            
            # Wait for trial duration
            start_time = time.time()
            while time.time() - start_time < self.trial_duration:
                if not self.is_running:
                    break
                while self.is_paused and self.is_running:
                    time.sleep(0.1)
                time.sleep(0.05)
            
            # Score responses if we're past n-back threshold
            if trial >= self.n_level:
                self.score_trial(trial)
            
            # Update display
            self.root.after(0, self.update_display)
            
            # Inter-trial interval
            self.root.after(0, self.clear_grid)
            
            time.sleep(self.inter_trial_interval)
        
        # Test completed
        if self.is_running:
            self.root.after(0, self.test_completed)
    
    def score_trial(self, trial_index):
        # Check if current trial matches n trials back
        n_back_position = self.position_history[trial_index - self.n_level]
        n_back_audio = self.audio_history[trial_index - self.n_level]
        
        current_position = self.position_history[trial_index]
        current_audio = self.audio_history[trial_index]
        
        # Position scoring
        position_match = (current_position == n_back_position)
        position_response = self.position_responses[trial_index]
        
        if position_match and position_response:
            self.position_correct += 1  # Hit
        elif position_match and not position_response:
            pass  # Miss
        elif not position_match and position_response:
            pass  # False alarm
        # else: Correct rejection (no response when no match)
        
        if position_match or position_response:
            self.position_total += 1
            if position_match and position_response:
                pass  # Already counted as correct
            elif position_match and not position_response:
                pass  # Miss - total incremented but not correct
            elif not position_match and position_response:
                pass  # False alarm - total incremented but not correct
        
        # Audio scoring
        audio_match = (current_audio == n_back_audio)
        audio_response = self.audio_responses[trial_index]
        
        if audio_match and audio_response:
            self.audio_correct += 1  # Hit
        elif audio_match and not audio_response:
            pass  # Miss
        elif not audio_match and audio_response:
            pass  # False alarm
        # else: Correct rejection
        
        if audio_match or audio_response:
            self.audio_total += 1
            if audio_match and audio_response:
                pass  # Already counted as correct
            elif audio_match and not audio_response:
                pass  # Miss - total incremented but not correct
            elif not audio_match and audio_response:
                pass  # False alarm - total incremented but not correct
    
    def update_display(self):
        # Update trial counter
        self.trial_label.config(text=f"Trial: {self.current_trial}/{self.num_trials}")
        
        # Update progress bar
        progress = (self.current_trial / self.num_trials) * 100
        self.progress_bar['value'] = progress
        
        # Update scores
        pos_percent = (self.position_correct / max(1, self.position_total)) * 100
        audio_percent = (self.audio_correct / max(1, self.audio_total)) * 100
        
        self.position_score_label.config(
            text=f"Position: {self.position_correct}/{self.position_total} ({pos_percent:.1f}%)")
        self.audio_score_label.config(
            text=f"Audio: {self.audio_correct}/{self.audio_total} ({audio_percent:.1f}%)")
    
    def test_completed(self):
        self.is_running = False
        self.clear_grid()
        
        # Calculate final scores
        pos_percent = (self.position_correct / max(1, self.position_total)) * 100
        audio_percent = (self.audio_correct / max(1, self.audio_total)) * 100
        
        # Check for level advancement
        level_up = False
        if pos_percent >= 80 and audio_percent >= 80:
            if self.n_level < 9:  # Max level
                self.n_level += 1
                self.level_var.set(str(self.n_level))
                level_up = True
        
        # Show results
        result_msg = f"""Test Completed!
        
N-Back Level: {self.n_level - (1 if level_up else 0)}

Results:
Position: {self.position_correct}/{self.position_total} ({pos_percent:.1f}%)
Audio: {self.audio_correct}/{self.audio_total} ({audio_percent:.1f}%)

{"🎉 Level Up! New level: " + str(self.n_level) if level_up else ""}
{"Need 80%+ on both to advance" if not level_up else ""}"""

        messagebox.showinfo("Test Results", result_msg)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import pyttsx3
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install pyttsx3")
        exit(1)
    
    app = DualNBackTest()
    app.run()
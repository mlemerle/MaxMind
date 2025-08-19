#!/usr/bin/env python3
"""
Clean Dual N-Back Implementation
Standalone pygame-based dual n-back cognitive training game.
"""
import pygame
import pygame.mixer
import random
import time
import os
import csv
import pyttsx3
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 3
CELL_SIZE = 90
GRID_GAP = 4  # Small gap between cells for clean separation
GRID_AREA_WIDTH = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GRID_GAP + 40  # Extra padding
CONTROL_PANEL_WIDTH = 250
WINDOW_WIDTH = GRID_AREA_WIDTH + CONTROL_PANEL_WIDTH + 30
WINDOW_HEIGHT = 500

# Colors - Apple Dark Mode Aesthetic
DARK_BG = (28, 28, 30)          # iOS dark background
CARD_BG = (44, 44, 46)          # iOS card background
ELEVATED_BG = (58, 58, 60)      # Elevated surfaces
WHITE = (255, 255, 255)
OFF_WHITE = (242, 242, 247)     # iOS label color
SECONDARY_WHITE = (174, 174, 178) # iOS secondary label
ACCENT_BLUE = (10, 132, 255)    # iOS blue
SUCCESS_GREEN = (48, 209, 88)   # iOS green
WARNING_ORANGE = (255, 149, 0)  # iOS orange
DESTRUCTIVE_RED = (255, 69, 58) # iOS red
SEPARATOR = (84, 84, 88)        # iOS separator

@dataclass
class GameConfig:
    n: int
    trials: int
    stimulus_ms: int
    isi_ms: int
    seed: Optional[int] = None

@dataclass
class Stimulus:
    pos: int  # 0-8 for 3x3 grid
    letter: str

@dataclass
class TrialResult:
    trial: int
    n: int
    pos: int
    letter: str
    pos_match_truth: bool
    letter_match_truth: bool
    pos_key_pressed: bool
    letter_key_pressed: bool
    pos_rt_ms: float
    letter_rt_ms: float
    correct_pos: bool
    correct_letter: bool
    seed: int

class AudioManager:
    def __init__(self):
        self.sounds = {}
        self.letters = ['C', 'H', 'K', 'L', 'Q', 'R', 'S', 'T']
        self.audio_dir = "assets/audio/letters"
        self.ensure_audio_files()
        self.load_sounds()
    
    def ensure_audio_files(self):
        """Generate audio files if they don't exist."""
        os.makedirs(self.audio_dir, exist_ok=True)
        
        for letter in self.letters:
            path = os.path.join(self.audio_dir, f"{letter}.wav")
            if not os.path.exists(path):
                self.generate_audio_file(letter, path)
    
    def generate_audio_file(self, letter, path):
        """Generate a single audio file using pyttsx3."""
        try:
            engine = pyttsx3.init()
            engine.save_to_file(letter, path)
            engine.runAndWait()
        except Exception as e:
            print(f"Could not generate audio for {letter}: {e}")
    
    def load_sounds(self):
        """Load all sound files into memory."""
        for letter in self.letters:
            path = os.path.join(self.audio_dir, f"{letter}.wav")
            if os.path.exists(path):
                try:
                    self.sounds[letter] = pygame.mixer.Sound(path)
                except pygame.error:
                    print(f"Could not load sound for {letter}")
    
    def play_letter(self, letter):
        """Play a letter sound."""
        if letter in self.sounds:
            self.sounds[letter].play()

class StimulusGenerator:
    def __init__(self, config: GameConfig):
        self.config = config
        self.letters = ['C', 'H', 'K', 'L', 'Q', 'R', 'S', 'T']
        self.stimuli = []
        if config.seed is not None:
            random.seed(config.seed)
        self.generate_sequence()
    
    def generate_sequence(self):
        """Pre-generate the entire stimulus sequence."""
        for trial in range(self.config.trials):
            pos = random.randint(0, 8)
            letter = random.choice(self.letters)
            self.stimuli.append(Stimulus(pos, letter))

class Button:
    def __init__(self, x, y, width, height, text, color=CARD_BG, text_color=OFF_WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 20)
        self.clicked = False
    
    def draw(self, screen):
        # Draw simple rectangle background
        pygame.draw.rect(screen, self.color, self.rect)
        # Minimal border
        pygame.draw.rect(screen, SEPARATOR, self.rect, 1)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                return True
        return False

class DualNBackGame:
    def __init__(self, config: GameConfig):
        self.config = config
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Dual N-Back Enhanced")
        self.clock = pygame.time.Clock()
        
        self.audio_manager = AudioManager()
        self.stimulus_generator = StimulusGenerator(config)
        
        # Game state
        self.state = "READY"  # READY, RUNNING, PAUSED, SUMMARY
        self.current_trial = 0
        self.trial_start_time = 0
        self.stimulus_active = False
        self.trial_results = []
        
        # Input tracking
        self.pos_key_pressed = False
        self.letter_key_pressed = False
        self.pos_rt_ms = -1
        self.letter_rt_ms = -1
        
        # Feedback tracking
        self.last_trial_feedback = {"pos": "", "letter": ""}
        
        # UI elements
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)

        # Create buttons
        self.buttons = {
            "start": Button(GRID_AREA_WIDTH + 20, 50, 200, 40, "Start Game", SUCCESS_GREEN),
            "pause": Button(GRID_AREA_WIDTH + 20, 100, 200, 40, "Pause", WARNING_ORANGE),
            "stop": Button(GRID_AREA_WIDTH + 20, 150, 200, 40, "Stop", DESTRUCTIVE_RED),
            "reset": Button(GRID_AREA_WIDTH + 20, 200, 200, 40, "Reset", ACCENT_BLUE)
        }
    
    def draw_grid(self):
        """Draw the 3x3 grid."""
        grid_start_x = 20
        grid_start_y = 100
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = grid_start_x + j * (CELL_SIZE + GRID_GAP)
                y = grid_start_y + i * (CELL_SIZE + GRID_GAP)
                
                # Determine if this cell should be highlighted
                highlighted = False
                if (self.state == "RUNNING" and self.stimulus_active and 
                    self.current_trial < len(self.stimulus_generator.stimuli)):
                    current_stimulus = self.stimulus_generator.stimuli[self.current_trial]
                    if current_stimulus.pos == i * GRID_SIZE + j:
                        highlighted = True
                
                # Draw cell
                color = ACCENT_BLUE if highlighted else CARD_BG
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, SEPARATOR, (x, y, CELL_SIZE, CELL_SIZE), 2)
    
    def draw_control_panel(self):
        """Draw the control panel."""
        # Title
        title = self.font.render("Dual N-Back", True, OFF_WHITE)
        self.screen.blit(title, (GRID_AREA_WIDTH + 20, 10))
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)
        
        # Game info
        info_y = 270
        info_texts = [
            f"Level: {self.config.n}-back",
            f"Trial: {self.current_trial + 1}/{self.config.trials}",
            f"State: {self.state}",
            "",
            "Controls:",
            "F - Position match",
            "J - Letter match",
            "ESC - Quit"
        ]
        
        for i, text in enumerate(info_texts):
            color = OFF_WHITE if text else SECONDARY_WHITE
            rendered = self.small_font.render(text, True, color)
            self.screen.blit(rendered, (GRID_AREA_WIDTH + 20, info_y + i * 20))
        
        # Feedback
        if self.last_trial_feedback["pos"] or self.last_trial_feedback["letter"]:
            feedback_y = info_y + len(info_texts) * 20 + 20
            pos_text = f"Pos: {self.last_trial_feedback['pos']}"
            letter_text = f"Letter: {self.last_trial_feedback['letter']}"
            
            pos_color = SUCCESS_GREEN if "Correct" in self.last_trial_feedback["pos"] else DESTRUCTIVE_RED
            letter_color = SUCCESS_GREEN if "Correct" in self.last_trial_feedback["letter"] else DESTRUCTIVE_RED
            
            pos_surface = self.small_font.render(pos_text, True, pos_color)
            letter_surface = self.small_font.render(letter_text, True, letter_color)
            
            self.screen.blit(pos_surface, (GRID_AREA_WIDTH + 20, feedback_y))
            self.screen.blit(letter_surface, (GRID_AREA_WIDTH + 20, feedback_y + 20))
    
    def draw_instructions(self):
        """Draw initial instructions."""
        instructions = [
            "Welcome to Dual N-Back Training",
            "",
            "1. Watch the grid and listen to letters",
            "2. Press F if position matches N trials back",
            "3. Press J if letter matches N trials back",
            "4. You can press both keys for dual matches",
            "",
            "Click START to begin training"
        ]
        
        start_y = 150
        for i, line in enumerate(instructions):
            color = OFF_WHITE if line else SECONDARY_WHITE
            text = self.font.render(line, True, color) if i == 0 else self.small_font.render(line, True, color)
            text_rect = text.get_rect(center=(GRID_AREA_WIDTH // 2, start_y + i * 25))
            self.screen.blit(text, text_rect)
    
    def draw_summary(self):
        """Draw results summary."""
        if not self.trial_results:
            return
            
        # Calculate statistics
        total_trials = len(self.trial_results)
        pos_correct = sum(1 for r in self.trial_results if r.correct_pos)
        letter_correct = sum(1 for r in self.trial_results if r.correct_letter)
        
        pos_accuracy = (pos_correct / total_trials) * 100
        letter_accuracy = (letter_correct / total_trials) * 100
        
        summary_lines = [
            "Training Complete!",
            "",
            f"Trials: {total_trials}",
            f"Position Accuracy: {pos_accuracy:.1f}%",
            f"Letter Accuracy: {letter_accuracy:.1f}%",
            "",
            "Results saved to CSV file"
        ]
        
        start_y = 150
        for i, line in enumerate(summary_lines):
            color = OFF_WHITE if line else SECONDARY_WHITE
            text = self.font.render(line, True, color) if i == 0 else self.small_font.render(line, True, color)
            text_rect = text.get_rect(center=(GRID_AREA_WIDTH // 2, start_y + i * 25))
            self.screen.blit(text, text_rect)
    
    def handle_trial_input(self, key):
        """Handle keyboard input during trials."""
        current_time = pygame.time.get_ticks()
        
        if key == pygame.K_f and not self.pos_key_pressed:
            self.pos_key_pressed = True
            self.pos_rt_ms = current_time - self.trial_start_time
        
        if key == pygame.K_j and not self.letter_key_pressed:
            self.letter_key_pressed = True
            self.letter_rt_ms = current_time - self.trial_start_time
    
    def evaluate_trial(self):
        """Evaluate the current trial and record results."""
        if self.current_trial < self.config.n:
            # Not enough history for n-back comparison
            pos_match_truth = False
            letter_match_truth = False
        else:
            current_stimulus = self.stimulus_generator.stimuli[self.current_trial]
            n_back_stimulus = self.stimulus_generator.stimuli[self.current_trial - self.config.n]
            
            pos_match_truth = current_stimulus.pos == n_back_stimulus.pos
            letter_match_truth = current_stimulus.letter == n_back_stimulus.letter
        
        # Determine correctness
        correct_pos = (self.pos_key_pressed and pos_match_truth) or (not self.pos_key_pressed and not pos_match_truth)
        correct_letter = (self.letter_key_pressed and letter_match_truth) or (not self.letter_key_pressed and not letter_match_truth)
        
        # Record result
        current_stimulus = self.stimulus_generator.stimuli[self.current_trial]
        result = TrialResult(
            trial=self.current_trial + 1,
            n=self.config.n,
            pos=current_stimulus.pos,
            letter=current_stimulus.letter,
            pos_match_truth=pos_match_truth,
            letter_match_truth=letter_match_truth,
            pos_key_pressed=self.pos_key_pressed,
            letter_key_pressed=self.letter_key_pressed,
            pos_rt_ms=self.pos_rt_ms if self.pos_key_pressed else -1,
            letter_rt_ms=self.letter_rt_ms if self.letter_key_pressed else -1,
            correct_pos=correct_pos,
            correct_letter=correct_letter,
            seed=self.config.seed or 0
        )
        
        self.trial_results.append(result)
        
        # Update feedback
        self.last_trial_feedback = {
            "pos": "Correct" if correct_pos else "Incorrect",
            "letter": "Correct" if correct_letter else "Incorrect"
        }
    
    def start_trial(self):
        """Start a new trial."""
        if self.current_trial >= self.config.trials:
            self.end_game()
            return
        
        # Reset trial state
        self.pos_key_pressed = False
        self.letter_key_pressed = False
        self.pos_rt_ms = -1
        self.letter_rt_ms = -1
        
        # Start stimulus
        self.trial_start_time = pygame.time.get_ticks()
        self.stimulus_active = True
        
        # Play audio
        current_stimulus = self.stimulus_generator.stimuli[self.current_trial]
        self.audio_manager.play_letter(current_stimulus.letter)
    
    def update_trial(self):
        """Update the current trial state."""
        if not self.stimulus_active:
            return
            
        current_time = pygame.time.get_ticks()
        
        # Check if stimulus period is over
        if current_time - self.trial_start_time >= self.config.stimulus_ms:
            self.stimulus_active = False
        
        # Check if trial is complete
        if current_time - self.trial_start_time >= self.config.isi_ms:
            self.evaluate_trial()
            self.current_trial += 1
            self.start_trial()
    
    def start_game(self):
        """Start the game."""
        self.state = "RUNNING"
        self.current_trial = 0
        self.trial_results = []
        self.start_trial()
    
    def pause_game(self):
        """Pause/unpause the game."""
        if self.state == "RUNNING":
            self.state = "PAUSED"
        elif self.state == "PAUSED":
            self.state = "RUNNING"
    
    def stop_game(self):
        """Stop the game and show summary."""
        if self.trial_results:
            self.save_results()
        self.state = "SUMMARY"
    
    def end_game(self):
        """End the game naturally."""
        self.save_results()
        self.state = "SUMMARY"
    
    def save_results(self):
        """Save results to CSV file."""
        if not self.trial_results:
            return
            
        # Create runs directory
        os.makedirs("runs", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"runs/run_{timestamp}.csv"
        
        # Write CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'trial', 'n', 'pos', 'letter', 'pos_match_truth', 'letter_match_truth',
                'pos_key_pressed', 'letter_key_pressed', 'pos_rt_ms', 'letter_rt_ms',
                'correct_pos', 'correct_letter', 'seed'
            ]
            writer.writerow(header)
            
            # Data
            for result in self.trial_results:
                row = [
                    result.trial, result.n, result.pos, result.letter,
                    result.pos_match_truth, result.letter_match_truth,
                    result.pos_key_pressed, result.letter_key_pressed,
                    result.pos_rt_ms, result.letter_rt_ms,
                    result.correct_pos, result.correct_letter, result.seed
                ]
                writer.writerow(row)
        
        print(f"Results saved to {filename}")
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.state == "RUNNING":
                        self.handle_trial_input(event.key)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle button clicks
                    if self.buttons["start"].handle_event(event):
                        if self.state in ["READY", "SUMMARY"]:
                            self.start_game()
                    
                    elif self.buttons["pause"].handle_event(event):
                        if self.state in ["RUNNING", "PAUSED"]:
                            self.pause_game()
                    
                    elif self.buttons["stop"].handle_event(event):
                        if self.state in ["RUNNING", "PAUSED"]:
                            self.stop_game()
                    
                    elif self.buttons["reset"].handle_event(event):
                        self.state = "READY"
                        self.current_trial = 0
                        self.trial_results = []
                        self.last_trial_feedback = {"pos": "", "letter": ""}
            
            # Update game state
            if self.state == "RUNNING":
                self.update_trial()
            
            # Draw everything
            self.screen.fill(DARK_BG)
            
            self.draw_grid()
            self.draw_control_panel()
            
            if self.state == "READY":
                self.draw_instructions()
            elif self.state == "SUMMARY":
                self.draw_summary()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Dual N-Back Test")
    parser.add_argument("--n", type=int, default=2, help="N-back level")
    parser.add_argument("--trials", type=int, default=40, help="Number of trials")
    parser.add_argument("--stimulus_ms", type=int, default=500, help="Stimulus duration in ms")
    parser.add_argument("--isi_ms", type=int, default=2500, help="Inter-stimulus interval in ms")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n >= args.trials:
        print("Error: N-back level must be less than number of trials")
        return
    
    config = GameConfig(
        n=args.n,
        trials=args.trials,
        stimulus_ms=args.stimulus_ms,
        isi_ms=args.isi_ms,
        seed=args.seed
    )
    
    game = DualNBackGame(config)
    game.run()

if __name__ == "__main__":
    main()

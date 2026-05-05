# Wordle Clone Development Plan

## Objective
Build a functional Wordle clone game with core mechanics: 5-letter word guessing, 6 attempts, color-coded feedback (green for correct position, yellow for wrong position, gray for not in word), and win/loss states.

## Files Involved
- `wordle.py` - Main game logic and pygame rendering
- `requirements.txt` - Dependencies (pygame, random, string)
- `README.md` - Setup and usage instructions
- `words.txt` - List of valid 5-letter words for validation

## Step-by-Step Implementation

1. **Setup Phase**
   - Create `words.txt` with 100-200 common 5-letter words
   - Write `requirements.txt` with pygame dependency
   - Draft `README.md` with installation and play instructions

2. **Core Game Logic**
   - Implement word selection (random from words.txt)
   - Create letter input system (keyboard or on-screen)
   - Build guess validation (check if guess is in word list)
   - Implement feedback algorithm (compare guess vs target word)

3. **Visual Rendering**
   - Design grid layout for 6 rows × 5 columns
   - Style tiles with color states (gray, yellow, green)
   - Add animations for tile flips and color changes
   - Display current guess and keyboard layout

4. **Game States**
   - Handle win condition (all tiles green)
   - Handle loss condition (6 attempts exhausted)
   - Show correct answer on loss
   - Add restart functionality

5. **Polish**
   - Add keyboard shortcuts (Enter to submit, Backspace to delete)
   - Include sound effects (optional)
   - Add simple animations for better UX

## Risks or Open Questions
- Word list source: Need to decide between embedding words or fetching from API
- Keyboard layout: Physical keyboard mapping vs on-screen keyboard
- Animation complexity: Keep simple or add smooth transitions?
- Mobile compatibility: Should we support touch input?
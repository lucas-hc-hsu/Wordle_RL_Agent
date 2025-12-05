"""
Render a browser-style Wordle game demo using Manim.

This creates a demo GIF showing the browser-based Wordle game interface,
complete with the 6x5 grid and on-screen keyboard.

Usage:
    source .venv-manim/bin/activate
    manim -pql scripts/rendering/render_browser_demo.py BrowserWordleDemo

Output:
    assets/wordle_browser_demo.gif
"""

from manim import config
import os
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.media_dir = str(OUTPUT_DIR)
config.video_dir = str(OUTPUT_DIR)

# Configure for browser-like aspect ratio (portrait-ish, like mobile)
config.frame_width = 9.0
config.frame_height = 16.0
config.pixel_width = 1080
config.pixel_height = 1920

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
import numpy as np

# =============================================================================
# Wordle Color Scheme (matching play_wordle.py)
# =============================================================================
WORDLE_GREEN = "#538d4e"   # Correct letter, correct position
WORDLE_YELLOW = "#b59f3b"  # Correct letter, wrong position
WORDLE_GREY = "#3a3a3c"    # Letter not in word
WORDLE_DARK = "#121213"    # Background
WORDLE_OUTLINE = "#3a3a3c" # Tile outline
WORDLE_WHITE = "#ffffff"   # Text color
WORDLE_KEY_BG = "#818384"  # Keyboard key background


class BrowserWordleDemo(Scene):
    """Render a browser-style Wordle game demo."""

    def construct(self):
        # Set background
        self.camera.background_color = WORDLE_DARK

        # Create header
        header = self.create_header()
        header.to_edge(UP, buff=0.5)

        # Create game board
        board = self.create_board()
        board.move_to(ORIGIN).shift(UP * 1.5)

        # Create keyboard
        keyboard = self.create_keyboard()
        keyboard.to_edge(DOWN, buff=1.0)

        # Create stats text
        stats = Text(
            "Enter to submit | Space for new game",
            font_size=20,
            color="#818384"
        )
        stats.next_to(keyboard, DOWN, buff=0.3)

        # Add all elements
        self.add(header, board, keyboard, stats)

        # Demo gameplay: CRANE -> STARE -> SHARE (win!)
        target_word = "SHARE"
        guesses = ["CRANE", "STARE", "SHARE"]

        # Animate the gameplay
        for guess_idx, guess in enumerate(guesses):
            # Type letters with animation
            for letter_idx, letter in enumerate(guess):
                tile = board[guess_idx][letter_idx]
                letter_text = Text(letter, font_size=48, color=WORDLE_WHITE, weight=BOLD)
                letter_text.move_to(tile.get_center())

                self.play(
                    FadeIn(letter_text, scale=1.2),
                    tile.animate.set_stroke(color="#565758"),
                    run_time=0.15
                )

            self.wait(0.3)

            # Get feedback
            feedback = self.get_feedback(guess, target_word)

            # Reveal feedback with flip animation
            for letter_idx, (letter, fb) in enumerate(zip(guess, feedback)):
                tile = board[guess_idx][letter_idx]
                color = self.get_color(fb)

                # Find the letter text (last added at this position)
                letter_text = Text(letter, font_size=48, color=WORDLE_WHITE, weight=BOLD)
                letter_text.move_to(tile.get_center())

                self.play(
                    tile.animate.set_fill(color=color, opacity=1),
                    tile.animate.set_stroke(color=color),
                    run_time=0.2
                )

                # Update keyboard
                self.update_keyboard_key(keyboard, letter, fb)

            self.wait(0.5)

        # Show win message
        win_msg = self.create_message("Impressive!")
        win_msg.move_to(ORIGIN).shift(UP * 5)
        self.play(FadeIn(win_msg, shift=DOWN * 0.5), run_time=0.3)

        self.wait(1.5)

        # Convert to GIF
        self.convert_to_gif()

    def create_header(self):
        """Create the WORDLE header."""
        title = Text("WORDLE", font_size=56, color=WORDLE_WHITE, weight=BOLD)
        title.set_stroke(width=0)

        subtitle = Text(
            "Wordle RL Agent - Play Mode",
            font_size=18,
            color="#818384"
        )
        subtitle.next_to(title, DOWN, buff=0.15)

        header = VGroup(title, subtitle)

        # Add bottom border line
        line = Line(
            start=LEFT * 4, end=RIGHT * 4,
            color="#3a3a3c", stroke_width=1
        )
        line.next_to(header, DOWN, buff=0.3)

        return VGroup(header, line)

    def create_board(self):
        """Create the 6x5 game board."""
        board = VGroup()
        tile_size = 0.9
        gap = 0.1

        for row in range(6):
            row_group = VGroup()
            for col in range(5):
                tile = Square(
                    side_length=tile_size,
                    fill_opacity=0,
                    stroke_color=WORDLE_OUTLINE,
                    stroke_width=2
                )
                x = (col - 2) * (tile_size + gap)
                y = (2.5 - row) * (tile_size + gap)
                tile.move_to([x, y, 0])
                row_group.add(tile)
            board.add(row_group)

        return board

    def create_keyboard(self):
        """Create the on-screen keyboard."""
        keyboard = VGroup()
        rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            ["ENT"] + list("ZXCVBNM") + ["⌫"]
        ]

        key_width = 0.65
        key_height = 0.85
        gap = 0.08

        for row_idx, row in enumerate(rows):
            row_group = VGroup()
            total_width = sum(
                key_width * 1.5 if k in ["ENT", "⌫"] else key_width
                for k in row
            ) + gap * (len(row) - 1)

            x_offset = -total_width / 2

            for key in row:
                is_wide = key in ["ENT", "⌫"]
                w = key_width * 1.5 if is_wide else key_width

                key_rect = RoundedRectangle(
                    width=w,
                    height=key_height,
                    corner_radius=0.08,
                    fill_color=WORDLE_KEY_BG,
                    fill_opacity=1,
                    stroke_width=0
                )

                display_key = "ENTER" if key == "ENT" else key
                font_size = 18 if is_wide else 24
                key_text = Text(
                    display_key,
                    font_size=font_size,
                    color=WORDLE_WHITE,
                    weight=BOLD
                )
                key_text.move_to(key_rect.get_center())

                key_group = VGroup(key_rect, key_text)
                key_group.move_to([x_offset + w / 2, 0, 0])
                key_group.key_letter = key if len(key) == 1 else None

                row_group.add(key_group)
                x_offset += w + gap

            row_group.shift(DOWN * row_idx * (key_height + gap))
            keyboard.add(row_group)

        return keyboard

    def create_message(self, text):
        """Create a message box."""
        msg_text = Text(text, font_size=28, color=WORDLE_DARK, weight=BOLD)
        msg_bg = RoundedRectangle(
            width=msg_text.width + 0.6,
            height=msg_text.height + 0.4,
            corner_radius=0.1,
            fill_color=WORDLE_WHITE,
            fill_opacity=1,
            stroke_width=0
        )
        msg_text.move_to(msg_bg.get_center())
        return VGroup(msg_bg, msg_text)

    def get_feedback(self, guess, target):
        """Get Wordle feedback for a guess."""
        guess = guess.upper()
        target = target.upper()
        feedback = ["grey"] * 5
        target_chars = list(target)

        # First pass: find greens
        for i in range(5):
            if guess[i] == target[i]:
                feedback[i] = "green"
                target_chars[i] = None

        # Second pass: find yellows
        for i in range(5):
            if feedback[i] != "green" and guess[i] in target_chars:
                feedback[i] = "yellow"
                target_chars[target_chars.index(guess[i])] = None

        return feedback

    def get_color(self, feedback):
        """Convert feedback to color."""
        return {
            "green": WORDLE_GREEN,
            "yellow": WORDLE_YELLOW,
            "grey": WORDLE_GREY
        }[feedback]

    def update_keyboard_key(self, keyboard, letter, feedback):
        """Update keyboard key color based on feedback."""
        color = self.get_color(feedback)
        for row in keyboard:
            for key_group in row:
                if hasattr(key_group, 'key_letter') and key_group.key_letter == letter:
                    key_rect = key_group[0]
                    # Simply update the key color
                    self.play(
                        key_rect.animate.set_fill(color=color),
                        run_time=0.1
                    )
                    return  # Only update once

    def convert_to_gif(self):
        """Convert the rendered video to GIF."""
        import subprocess
        import shutil

        # Find the rendered video
        video_dir = Path(config.video_dir) / "videos" / "1080p60"
        if not video_dir.exists():
            video_dir = Path(config.video_dir) / "videos" / "480p15"

        video_files = list(video_dir.glob("*.mp4")) if video_dir.exists() else []

        if not video_files:
            print("No video file found to convert")
            return

        video_path = video_files[-1]
        gif_path = OUTPUT_DIR / "wordle_browser_demo.gif"

        print(f"Converting {video_path} to {gif_path}")

        # Use ffmpeg to create optimized GIF
        palette_path = "/tmp/palette.png"

        # Generate palette
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=15,scale=540:-1:flags=lanczos,palettegen=stats_mode=diff",
            palette_path
        ], capture_output=True)

        # Create GIF with palette
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path), "-i", palette_path,
            "-lavfi", "fps=15,scale=540:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            str(gif_path)
        ], capture_output=True)

        if gif_path.exists():
            size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"GIF created: {gif_path} ({size_mb:.2f} MB)")
        else:
            print("GIF creation failed")


if __name__ == "__main__":
    # For testing
    scene = BrowserWordleDemo()
    scene.construct()

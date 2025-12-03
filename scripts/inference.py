"""
Interactive Wordle inference script for playing with a trained agent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils.word_list import load_word_list


class WordlePlayer:
    def __init__(self, agent, word_list):
        self.agent = agent
        self.word_list = word_list
        self.current_state = np.zeros(26 * 3)
        self.possible_words = word_list.copy()

    def initial_guess(self):
        action = self.agent.act(self.current_state)
        return self.word_list[action]

    def update_state(self, guess, feedback):
        """
        Update state based on user feedback.
        feedback: string of length 5, each character represents the color at that position
        'g': green (correct position)
        'y': yellow (wrong position)
        'b': gray (not in word)
        """
        new_possible_words = []

        # Update state vector
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            idx = ord(letter) - ord("a")
            if color == "g":
                self.current_state[idx * 3 + 1] = 1
            elif color == "y":
                self.current_state[idx * 3 + 2] = 1
            else:  # 'b'
                self.current_state[idx * 3] = 1

        # Filter possible words
        for word in self.possible_words:
            if self._is_word_possible(word, guess, feedback):
                new_possible_words.append(word)

        self.possible_words = new_possible_words

    def _is_word_possible(self, word, guess, feedback):
        for i, (g_letter, color) in enumerate(zip(guess, feedback)):
            if color == "g" and word[i] != g_letter:
                return False
            elif color == "y":
                if g_letter not in word or word[i] == g_letter:
                    return False
            elif color == "b" and g_letter in word:
                # Check if this letter is marked as yellow or green in other positions
                other_positions = [
                    j
                    for j, (_, c) in enumerate(zip(guess, feedback))
                    if j != i and (c == "y" or c == "g") and guess[j] == g_letter
                ]
                if not other_positions:
                    return False
        return True

    def next_guess(self):
        if not self.possible_words:
            return None
        action = self.agent.act(self.current_state)
        return self.word_list[action]


def play_wordle():
    # Load trained agent and word list
    word_list = load_word_list()
    env = WordleEnv(word_list)
    agent = WordleAgent(env.state_size, len(word_list), word_list)
    # Load trained model weights here
    # agent.model.load_state_dict(torch.load('path_to_model_weights.pth'))

    player = WordlePlayer(agent, word_list)

    # Initial guess
    guess = player.initial_guess()
    print(f"Initial guess: {guess}")

    for attempt in range(6):
        # Get user feedback
        feedback = input(
            f"Enter feedback for '{guess}' (g=green, y=yellow, b=black): "
        ).lower()

        if feedback == "ggggg":
            print("Congratulations! Word found!")
            break

        # Update state and get next guess
        player.update_state(guess, feedback)
        guess = player.next_guess()

        if guess is None:
            print("No possible words remaining!")
            break

        print(f"Next guess: {guess}")


if __name__ == "__main__":
    play_wordle()

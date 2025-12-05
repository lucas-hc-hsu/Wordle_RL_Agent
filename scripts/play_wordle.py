#!/usr/bin/env python3
"""
Wordle Web Simulator

A browser-based Wordle game that runs on localhost.
Uses the same word list as the RL agent for consistent gameplay.

Usage:
    python scripts/play_wordle.py

The game will automatically open in your default browser.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse

from utils.word_list import load_word_list_extended, load_common_words

# Configuration
HOST = "localhost"
PORT = 8080

# Load word lists
# Use extended word list (Wordle + NLTK) for valid guesses - more words accepted
VALID_WORDS = set(word.upper() for word in load_word_list_extended())
# Use common everyday words as target words - easier to guess
TARGET_WORDS = [word.upper() for word in load_common_words()]


def get_feedback(guess: str, target: str) -> list:
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


# HTML template for the Wordle game
# Using string replacement instead of .format() to avoid CSS brace conflicts
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wordle RL Agent - Play Wordle</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Clear Sans', 'Helvetica Neue', Arial, sans-serif;
            background-color: #121213;
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        header {
            width: 100%;
            max-width: 500px;
            padding: 16px;
            border-bottom: 1px solid #3a3a3c;
            text-align: center;
        }

        h1 {
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 0.2rem;
        }

        .subtitle {
            font-size: 12px;
            color: #818384;
            margin-top: 4px;
        }

        #game-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            flex-grow: 1;
            padding: 20px;
            width: 100%;
            max-width: 500px;
        }

        #board {
            display: grid;
            grid-template-rows: repeat(6, 1fr);
            gap: 5px;
            padding: 10px;
        }

        .row {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 5px;
        }

        .tile {
            width: 62px;
            height: 62px;
            border: 2px solid #3a3a3c;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 32px;
            font-weight: bold;
            text-transform: uppercase;
            transition: transform 0.1s;
        }

        .tile.filled {
            border-color: #565758;
            animation: pop 0.1s;
        }

        .tile.green {
            background-color: #538d4e;
            border-color: #538d4e;
        }

        .tile.yellow {
            background-color: #b59f3b;
            border-color: #b59f3b;
        }

        .tile.grey {
            background-color: #3a3a3c;
            border-color: #3a3a3c;
        }

        @keyframes pop {
            50% { transform: scale(1.1); }
        }

        @keyframes flip {
            0% { transform: rotateX(0); }
            50% { transform: rotateX(90deg); }
            100% { transform: rotateX(0); }
        }

        .tile.flip {
            animation: flip 0.5s ease;
        }

        #keyboard {
            width: 100%;
            max-width: 500px;
            padding: 8px;
        }

        .keyboard-row {
            display: flex;
            justify-content: center;
            gap: 6px;
            margin-bottom: 8px;
        }

        .key {
            height: 58px;
            min-width: 43px;
            padding: 0 12px;
            border-radius: 4px;
            background-color: #818384;
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            text-transform: uppercase;
            transition: background-color 0.1s;
        }

        .key:hover {
            opacity: 0.8;
        }

        .key.wide {
            min-width: 65px;
            font-size: 12px;
        }

        .key.green {
            background-color: #538d4e;
        }

        .key.yellow {
            background-color: #b59f3b;
        }

        .key.grey {
            background-color: #3a3a3c;
        }

        #message {
            height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 16px;
            font-weight: bold;
        }

        .message-box {
            background-color: #ffffff;
            color: #121213;
            padding: 12px 16px;
            border-radius: 4px;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        #new-game-btn {
            margin-top: 16px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            background-color: #538d4e;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            display: none;
        }

        #new-game-btn:hover {
            opacity: 0.9;
        }

        #stats {
            margin-top: 8px;
            font-size: 12px;
            color: #818384;
        }
    </style>
</head>
<body>
    <header>
        <h1>WORDLE</h1>
        <div class="subtitle">Wordle RL Agent - Play Mode</div>
    </header>

    <div id="game-container">
        <div id="message"></div>

        <div id="board"></div>

        <div id="keyboard"></div>

        <button id="new-game-btn" onclick="newGame()">New Game</button>

        <div id="stats">Valid words: __WORD_COUNT__ | Enter to submit | Space for new game</div>
    </div>

    <script>
        const VALID_WORDS = __VALID_WORDS__;
        const validWordsSet = new Set(VALID_WORDS);

        let targetWord = "";
        let currentRow = 0;
        let currentTile = 0;
        let gameOver = false;
        let guesses = [];

        const keyboard = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '⌫']
        ];

        const keyStates = {};

        function initBoard() {
            const board = document.getElementById('board');
            board.innerHTML = '';
            for (let i = 0; i < 6; i++) {
                const row = document.createElement('div');
                row.className = 'row';
                for (let j = 0; j < 5; j++) {
                    const tile = document.createElement('div');
                    tile.className = 'tile';
                    tile.id = `tile-${i}-${j}`;
                    row.appendChild(tile);
                }
                board.appendChild(row);
            }
        }

        function initKeyboard() {
            const keyboardEl = document.getElementById('keyboard');
            keyboardEl.innerHTML = '';
            keyboard.forEach(row => {
                const rowEl = document.createElement('div');
                rowEl.className = 'keyboard-row';
                row.forEach(key => {
                    const keyEl = document.createElement('button');
                    keyEl.className = 'key' + (key.length > 1 ? ' wide' : '');
                    keyEl.textContent = key;
                    keyEl.id = `key-${key}`;
                    keyEl.onclick = () => handleKey(key);
                    rowEl.appendChild(keyEl);
                });
                keyboardEl.appendChild(rowEl);
            });
        }

        function handleKey(key) {
            if (gameOver) return;

            if (key === 'ENTER') {
                submitGuess();
            } else if (key === '⌫') {
                deleteLetter();
            } else if (currentTile < 5) {
                addLetter(key);
            }
        }

        function addLetter(letter) {
            const tile = document.getElementById(`tile-${currentRow}-${currentTile}`);
            tile.textContent = letter;
            tile.classList.add('filled');
            currentTile++;
        }

        function deleteLetter() {
            if (currentTile > 0) {
                currentTile--;
                const tile = document.getElementById(`tile-${currentRow}-${currentTile}`);
                tile.textContent = '';
                tile.classList.remove('filled');
            }
        }

        async function submitGuess() {
            if (currentTile !== 5) {
                showMessage('Not enough letters');
                return;
            }

            let guess = '';
            for (let i = 0; i < 5; i++) {
                guess += document.getElementById(`tile-${currentRow}-${i}`).textContent;
            }

            if (!validWordsSet.has(guess)) {
                showMessage('Not in word list');
                shakeRow();
                return;
            }

            // Get feedback from server
            const response = await fetch(`/guess?word=${guess}&target=${targetWord}`);
            const data = await response.json();
            const feedback = data.feedback;

            guesses.push(guess);

            // Reveal tiles with animation
            for (let i = 0; i < 5; i++) {
                setTimeout(() => {
                    const tile = document.getElementById(`tile-${currentRow}-${i}`);
                    tile.classList.add('flip', feedback[i]);
                    updateKeyState(guess[i], feedback[i]);
                }, i * 300);
            }

            setTimeout(() => {
                if (guess === targetWord) {
                    gameOver = true;
                    const messages = ['Genius!', 'Magnificent!', 'Impressive!', 'Splendid!', 'Great!', 'Phew!'];
                    showMessage(messages[currentRow]);
                    document.getElementById('new-game-btn').style.display = 'block';
                } else if (currentRow === 5) {
                    gameOver = true;
                    showMessage(targetWord);
                    document.getElementById('new-game-btn').style.display = 'block';
                } else {
                    currentRow++;
                    currentTile = 0;
                }
            }, 1500);
        }

        function updateKeyState(letter, state) {
            const key = document.getElementById(`key-${letter}`);
            if (!key) return;

            const currentState = keyStates[letter];
            if (state === 'green' || (state === 'yellow' && currentState !== 'green') ||
                (state === 'grey' && !currentState)) {
                keyStates[letter] = state;
                key.className = 'key ' + state;
            }
        }

        function showMessage(msg) {
            const messageEl = document.getElementById('message');
            messageEl.innerHTML = `<div class="message-box">${msg}</div>`;
            if (!gameOver) {
                setTimeout(() => { messageEl.innerHTML = ''; }, 1500);
            }
        }

        function shakeRow() {
            const row = document.getElementById('board').children[currentRow];
            row.style.animation = 'none';
            row.offsetHeight;
            row.style.animation = 'shake 0.5s';
        }

        async function newGame() {
            const response = await fetch('/new-game');
            const data = await response.json();
            targetWord = data.target;

            currentRow = 0;
            currentTile = 0;
            gameOver = false;
            guesses = [];

            Object.keys(keyStates).forEach(key => delete keyStates[key]);

            initBoard();
            initKeyboard();
            document.getElementById('message').innerHTML = '';
            document.getElementById('new-game-btn').style.display = 'none';
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === ' ') {
                e.preventDefault();
                newGame();
            } else if (e.key === 'Enter') {
                handleKey('ENTER');
            } else if (e.key === 'Backspace') {
                handleKey('⌫');
            } else if (/^[a-zA-Z]$/.test(e.key)) {
                handleKey(e.key.toUpperCase());
            }
        });

        // Initialize game
        newGame();
    </script>
</body>
</html>
"""


class WordleHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for Wordle game."""

    def __init__(self, *args, **kwargs):
        self.target_word = random.choice(TARGET_WORDS)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            html = HTML_TEMPLATE.replace(
                "__VALID_WORDS__", json.dumps(list(VALID_WORDS))
            ).replace(
                "__WORD_COUNT__", str(len(VALID_WORDS))
            )
            self.wfile.write(html.encode())

        elif parsed_path.path == "/new-game":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            target = random.choice(TARGET_WORDS)
            self.wfile.write(json.dumps({"target": target}).encode())

        elif parsed_path.path == "/guess":
            query = urllib.parse.parse_qs(parsed_path.query)
            word = query.get("word", [""])[0].upper()
            target = query.get("target", [""])[0].upper()

            feedback = get_feedback(word, target)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"feedback": feedback}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP request logs."""
        pass


def open_browser():
    """Open browser after a short delay."""
    import time
    time.sleep(0.5)
    try:
        webbrowser.open(f"http://{HOST}:{PORT}")
    except Exception:
        pass  # Browser open failed, user can manually navigate


def main():
    print("=" * 60)
    print("Wordle RL Agent - Play Wordle")
    print("=" * 60)
    print(f"\nServer running at: http://{HOST}:{PORT}")
    print(f"Valid words: {len(VALID_WORDS)}")
    print(f"Target words: {len(TARGET_WORDS)}")
    print("\nOpen http://localhost:8080 in your browser to play!")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    # Open browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    server = HTTPServer((HOST, PORT), WordleHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

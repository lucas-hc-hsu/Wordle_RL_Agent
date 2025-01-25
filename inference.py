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
        更新狀態基於用戶反饋
        feedback: 字串，長度為5，每個字符代表該位置的顏色
        'g': 綠色 (正確位置)
        'y': 黃色 (錯誤位置)
        'b': 灰色 (不在單詞中)
        """
        new_possible_words = []

        # 更新狀態向量
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            idx = ord(letter) - ord("a")
            if color == "g":
                self.current_state[idx * 3 + 1] = 1
            elif color == "y":
                self.current_state[idx * 3 + 2] = 1
            else:  # 'b'
                self.current_state[idx * 3] = 1

        # 過濾可能的單詞
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
                # 檢查是否這個字母在其他位置被標記為黃色或綠色
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
    # 加載訓練好的agent和詞表
    word_list = load_word_list()
    env = WordleEnv(word_list)
    agent = WordleAgent(env.state_size, len(word_list), word_list)
    # 這裡需要載入已訓練的模型權重
    # agent.model.load_state_dict(torch.load('path_to_model_weights.pth'))

    player = WordlePlayer(agent, word_list)

    # 初始猜測
    guess = player.initial_guess()
    print(f"Initial guess: {guess}")

    for attempt in range(6):
        # 獲取用戶反饋
        feedback = input(
            f"Enter feedback for '{guess}' (g=green, y=yellow, b=black): "
        ).lower()

        if feedback == "ggggg":
            print("Congratulations! Word found!")
            break

        # 更新狀態並獲取下一個猜測
        player.update_state(guess, feedback)
        guess = player.next_guess()

        if guess is None:
            print("No possible words remaining!")
            break

        print(f"Next guess: {guess}")


if __name__ == "__main__":
    play_wordle()

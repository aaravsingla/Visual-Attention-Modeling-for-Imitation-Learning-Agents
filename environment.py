import numpy as np

class ReadingEnv:
    def __init__(self, sentence_length=10):
        self.length = sentence_length
        # Generate random "difficulty" for words: High value = Hard word (needs fixation)
        self.word_difficulty = np.random.rand(sentence_length)
        self.reset()

    def reset(self):
        self.position = 0
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # State: [normalized_position, current_word_difficulty]
        if self.position >= self.length:
            return np.array([1.0, 0.0]) # End of sentence
        return np.array([self.position/self.length, self.word_difficulty[self.position]])

    def step(self, action):
        # Action 0: Fixate (Stay), Action 1: Saccade (Move Right)
        done = False
        if action == 1:
            self.position += 1

        if self.position >= self.length:
            done = True

        self.step_count += 1
        return self._get_state(), done

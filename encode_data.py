import numpy as np


class SMILESEncoder:
    def __init__(self) -> None:
        pass

        # Allowed tokens (adapted from default dictionary)
        self._tokens = np.sort([k for k in {'#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5',
                                            '6', '7', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S',
                                            '[', ']', 'c', 'l', 'n', 'o', 'r', 's', '\\', '\n'}])  # < , > for consistency

        self.image_template = np.zeros((100, len(self._tokens) + 1))  # +1 for unknown. 100 * number of unique chars

        self.c2i = {}
        self.i2c = {}

        for i, c in enumerate(self._tokens):
            self.c2i[c] = i
            self.i2c[i] = c

    def encode(self, data):

        assert len(data) < 100

        char_image_template = self.image_template[:, :]

        for i, c in enumerate(data):
            char_image_template[i][self.c2i[c]] = 1

        return np.array(char_image_template)

    def decode(self, one_hot):
        text = ""
        for row in one_hot:
            for i, elem in enumerate(row):
                if elem == 1 and self.i2c.get(i):
                    text += self.i2c.get(i)

        return text

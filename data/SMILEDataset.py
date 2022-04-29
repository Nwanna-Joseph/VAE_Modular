from torch.utils.data import Dataset


class SMILEDataset(Dataset):
    # Adapter pattern (Software design pattern)
    def __init__(self, file_path="./data/smiles.txt"):
        with open(file_path, 'r') as f:
            text = f.read()

        # get unique characters in the data
        unique_chars = set(text)

        # create a map of SMILE character to index
        hashmap_char_index = {}
        # create a map of SMILE index to character
        hashmap_index_char = {}

        for idx, ch in enumerate(unique_chars):
            hashmap_char_index[ch] = idx
            hashmap_index_char[idx] = ch

        # text is a concatenation of all the chemical structure. split them via the next line aka \n

        self.data = text.split('\n')
        # self.data[0] is now C[C@@]1(C(=O)C=C(O1)C(=O)[O-])c2ccccc2

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

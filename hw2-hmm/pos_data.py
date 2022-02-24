"""Dataclass."""

class POSData:
    """Dataclass for tagged parts-of-speech data."""
    END_STATE = "<END_STATE>"
    END_OBS = "<END_OBS>"
    def __init__(self, root, train, read=True, end_state=False):
        self.root = root
        self.train = train
        self.X = None
        self.y = None
        self.end_state = end_state
        if read:
            self.read_txt()

    def read_txt(self):
        """Note: The slash char '/' is the separator between words
        and tags, but may also appear within words in the text. 
        Slashes never appear in the tags; the separator is always 
        the last slash in the <word/tag> sequence. 
        
        tldr; use string.rsplit('/') as opposed to string.split('/').
        """
        with open(self.root, mode='r', encoding="UTF-8") as f:
            if self.train:
                self.X, self.y = [], []
                for line in f:
                    cur_X, cur_y = [], []
                    elle = line.split()
                    for word in elle:
                        word_tag = word.rsplit('/', maxsplit=1)
                        assert len(word_tag) == 2
                        assert '/' not in word_tag[1]
                        cur_X.append(word_tag[0])
                        cur_y.append(word_tag[1])
                    if self.end_state:
                        cur_X.append(self.END_OBS)
                        cur_y.append(self.END_STATE)
                    self.X.append(cur_X)
                    self.y.append(cur_y)
            else:
                self.X = []
                for line in f:
                    #self.X.append(cur_X)
                    cur_X = line.split()
                    if self.end_state:
                        cur_X.append(self.END_OBS)
                    self.X.append(cur_X)
    
    def __iter__(self):
        return zip(self.X, self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        """Returns number of *sequences* in data."""
        return len(self.X)

if __name__ == "__main__":
    print("testing iterator.")
    data = POSData(
            root="hmm-training-data/it_isdt_train_tagged.txt", 
            train=True, 
            read=True
        )
    for x, y in data:
        print(x, y)
        break
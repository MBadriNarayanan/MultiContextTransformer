import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SLT_Dataset(Dataset):
    def __init__(self, nlp, dataframe, word_dict):
        self.dataframe = dataframe
        self.word_dict = word_dict
        self.span8 = "Span8/span=8_stride=2/"
        self.span12 = "Span12/span=12_stride=2/"
        self.span16 = "Span16/span=16_stride=2/"
        self.nlp = nlp

    def __len__(self):
        return self.dataframe.shape[0]

    def _get_tokens(self, sentence):
        tokens = self.nlp(sentence)

        l = []
        for token in tokens:
            l.append(self.word_dict.get(token.text, self.word_dict["<OOV>"]))
        token_tensor = torch.FloatTensor(l)
        return token_tensor

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]["name"]
        translation = self.dataframe.iloc[idx]["translation"]
        translation_tokens = self._get_tokens(translation)
        span8_tensor = pad_sequence(torch.load(self.span8 + filename + ".pt"))
        span12_tensor = pad_sequence(torch.load(self.span12 + filename + ".pt"))
        span16_tensor = pad_sequence(torch.load(self.span16 + filename + ".pt"))

        return ((span8_tensor, span12_tensor, span16_tensor), translation_tokens)

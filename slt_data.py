import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SLT_Dataset(Dataset):
    def __init__(self, nlp, dataframe, word_dict, frame_drop_flag):
        self.dataframe = dataframe
        self.word_dict = word_dict
        self.span8 = "Span8/span=8_stride=2/"
        self.span12 = "Span12/span=12_stride=2/"
        self.span16 = "Span16/span=16_stride=2/"
        self.frame_drop_flag = frame_drop_flag
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

    def frame_drop(self, span_tensor, idx_drop):
        idx = []
        span_tensor = span_tensor[:, :-2, :]
        if span_tensor.shape[1] > 50:
            allowed_len = span_tensor.shape[1] - (span_tensor.shape[1] % 5)
            for i in range(allowed_len):
                if (i + 1) % 5 == idx_drop:
                    continue
                else:
                    idx.append(i)
            span_tensor = span_tensor[:, idx, :]
        return span_tensor

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]["name"]
        translation = self.dataframe.iloc[idx]["translation"]
        translation_tokens = self._get_tokens(translation)
        span8_tensor = pad_sequence(torch.load(self.span8 + filename + ".pt"))
        span12_tensor = pad_sequence(torch.load(self.span12 + filename + ".pt"))
        span16_tensor = pad_sequence(torch.load(self.span16 + filename + ".pt"))

        if self.frame_drop_flag != 0:
            if self.frame_drop_flag == 1:
                span8_tensor = self.frame_drop(span8_tensor, 0)
                span12_tensor = self.frame_drop(span12_tensor, 0)
                span16_tensor = self.frame_drop(span16_tensor, 0)
            else:
                span8_tensor = self.frame_drop(span8_tensor, 1)
                span12_tensor = self.frame_drop(span12_tensor, 0)
                span16_tensor = self.frame_drop(span16_tensor, 3)
        return ((span8_tensor, span12_tensor, span16_tensor), translation_tokens)

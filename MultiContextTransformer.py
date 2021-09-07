import torch

from torch.nn import Module, Linear, Embedding
from transformer import *


class MultiContextTransformer(Module):
    def __init__(
        self,
        vocab_size,
        dmodel_encoder,
        dmodel_decoder,
        nhid_encoder,
        nhid_decoder,
        nlayers_encoder,
        nlayers_decoder,
        concat_input,
        concat_output,
        nhead_encoder,
        nhead_decoder,
        dropout,
        activation,
        pretrained_embedding,
        embedding_matrix,
        device,
    ):
        super(MultiContextTransformer, self).__init__()
        encoderlayers = TransformerEncoderLayer(
            dmodel_encoder,
            nhead_encoder,
            dim_feedforward=nhid_encoder,
            dropout=dropout,
            activation=activation,
        )
        decoderlayers = TransformerDecoderLayer(
            dmodel_decoder,
            nhead_decoder,
            dim_feedforward=nhid_decoder,
            dropout=dropout,
            activation=activation,
        )
        self.device = device
        self.span8encoder = TransformerEncoder(encoderlayers, nlayers_encoder)
        self.span12encoder = TransformerEncoder(encoderlayers, nlayers_encoder)
        self.span16encoder = TransformerEncoder(encoderlayers, nlayers_encoder)

        if pretrained_embedding == True:
            print("Loaded pretrained Embedding Matrix in model!")
            self.decoder_embedding = Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix), freeze=False
            )
        else:
            print("Embedding Layer not pretrained!")
            self.decoder_embedding = Embedding(vocab_size, dmodel_decoder)
        self.decoder = TransformerDecoder(decoderlayers, nlayers_decoder)
        self.concatlinear = Linear(concat_input, concat_output)
        self.finallinear = Linear(concat_output, vocab_size)

        self.PositionalEncodingEncoder = PositionalEncoding(dmodel_encoder).to(
            self.device
        )
        self.PositionalEncodingDecoder = PositionalEncoding(dmodel_decoder).to(
            self.device
        )

    def create_target_mask(self, target):

        target_mask = (
            torch.triu(torch.ones(target.shape[1], target.shape[1])) == 1
        ).transpose(0, 1)

        target_mask = (
            target_mask.float()
            .masked_fill(target_mask == 0, float("-inf"))
            .masked_fill(target_mask == 1, float(0.0))
        )

        return target_mask

    def forward(self, span8src, span12src, span16src, targets):

        tgt_mask = self.create_target_mask(targets).to(self.device)

        span8src = self.PositionalEncodingEncoder(span8src)
        span12src = self.PositionalEncodingEncoder(span12src)
        span16src = self.PositionalEncodingEncoder(span16src)

        zvector8 = self.span8encoder(span8src)
        zvector12 = self.span12encoder(span12src)
        zvector16 = self.span16encoder(span16src)

        concat_zvector = torch.cat((zvector8, zvector12, zvector16), 2)
        concat_vector = self.concatlinear(concat_zvector)
        decoder_input = self.decoder_embedding(targets)
        decoder_input = decoder_input.to(self.device)
        decoder_input = decoder_input.permute(1, 0, 2)
        decoder_input = self.PositionalEncodingDecoder(decoder_input)
        decoder_output = self.decoder(decoder_input, concat_vector, tgt_mask)
        final_output = self.finallinear(decoder_output)
        final_output = final_output.permute(1, 2, 0)
        del tgt_mask
        del zvector8, zvector12, zvector16
        del span8src, span12src, span16src, targets
        del concat_zvector, concat_vector
        del decoder_input, decoder_output
        return final_output

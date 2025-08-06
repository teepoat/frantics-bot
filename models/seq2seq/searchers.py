import torch
import torch.nn as nn
from .constants import BOS_TOKEN


class GreedySearch(nn.Module):
    def __init__(self, encoder, decoder, embedding, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.device = device

    def forward(self, x, input_length, max_length):
        encoder_outputs, hidden = self.encoder(x, input_length)
        decoder_hidden = hidden[:self.decoder.num_layers]
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * BOS_TOKEN
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)

        for _ in range(max_length):
            decoder_outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_outputs, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input.unsqueeze_(0)

        return all_tokens, all_scores
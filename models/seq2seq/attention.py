import torch
import torch.nn.functional as F
import torch.nn as nn
from .custom_types import Method


class LuongAttention(nn.Module):
    def __init__(self, method: Method, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        if not isinstance(method, Method):
            raise ValueError(method, f"should be a member of `Method` enum")
        match method:
            case Method.DOT:
                self.method = self.dot
            case Method.GENERAL:
                self.method = self.general
                self.Wa = nn.Linear(hidden_size, hidden_size)
            case Method.CONCAT:
                self.method = self.concat
                self.Wa = nn.Linear(hidden_size * 2, hidden_size)
                self.Va = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def dot(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general(self, hidden, encoder_outputs):
        return torch.sum(hidden * self.Wa(encoder_outputs), dim=2)

    def concat(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2)
        energy = self.Wa(torch.cat((hidden.permute(1, 0, 2).expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)).tanh()
        return torch.sum(self.Va * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_weights = self.method(hidden, encoder_outputs)
        return F.softmax(attn_weights, dim=1).unsqueeze(1)
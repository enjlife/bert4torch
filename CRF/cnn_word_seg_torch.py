import torch.nn
from torch import nn
from crf_torch import CRF


class CnnWordSeg(nn.Module):
    def __init__(self, num_labels, vocab_size, hidden_size):
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.crf = CRF(num_tags=num_labels, batch_first=False)

    def forward(self, x, mask):
        hidden_state = self.embedding(x)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)
        hidden_state = self.crf(hidden_state)
        return hidden_state

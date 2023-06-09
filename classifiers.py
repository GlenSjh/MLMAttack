import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, num_labels=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, num_labels)


    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        lengths = attention_mask.sum(-1)
        if(input_ids is not None):
            inputs_embeds = self.embedding(input_ids)
        packed_inputs = pack_padded_sequence(inputs_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

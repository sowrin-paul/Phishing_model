import torch
import torch.nn as nn
import torch.nn.functional as f

class BiGRUPhishingDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BiGRUPhishingDetector, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)

        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # classification
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def attention_mechanism(self, gru_output, mask=None):
        attention_scores = self.attention(gru_output) # calculating attention score with batch size, seq_len
        attention_scores = attention_scores.squeeze(-1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weight = f.softmax(attention_scores, dim=1)

        weighted_output = torch.bmm(
            attention_weight.unsqueeze(1),
            gru_output
        )

        return weighted_output.squeeze(1), attention_weight

    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)

        # GRU
        gru_output, _ = self.gru(embedded)

        attended_output, attention_weight = self.attention_mechanism(gru_output, mask)

        output = self.dropout2(attended_output)
        output = f.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))

        return output.squeeze(-1), attention_weight
import torch.nn as nn

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, seq_len, padding_idx, kernel_size=3, num_filters=100, dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the sequence length after convolution and pooling
        seq_len_after_pooling = self.calculate_seq_len_after_pooling(seq_len, kernel_size, pool_size=2)
        
        # Fully connected layer input size after pooling
        self.fc = nn.Linear(num_filters * seq_len_after_pooling, num_classes)
        self.dropout = nn.Dropout(dropout)

    def calculate_seq_len_after_pooling(self, seq_len, kernel_size, pool_size=2):
        # Sequence length after convolution: (seq_len - kernel_size + 1)
        conv_out = seq_len - kernel_size + 1
        # Sequence length after pooling: (conv_out // pool_size)
        pooled_out = conv_out // pool_size
        return pooled_out

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.long()
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        x = self.conv1(x)  # [batch_size, num_filters, seq_len - kernel_size + 1]
        x = self.pool(x)  # [batch_size, num_filters, (seq_len - kernel_size + 1) // 2]
        
        # Flatten the tensor dynamically based on the output shape
        x = x.view(x.size(0), -1)  # Flatten the tensor: [batch_size, num_filters * pooled_length]
        
        x = self.dropout(x)
        x = self.fc(x)# [batch_size, num_classes]
        return x

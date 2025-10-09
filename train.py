import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TinyTransformerBlock

# Hyperparameters
seq_len, vocab_size, d_model = 10, 20, 32
num_heads, dim_ff = 4, 64
epochs = 200

# Model parts
embedding = nn.Embedding(vocab_size, d_model)
transformer = TinyTransformerBlock(d_model=d_model, num_heads=num_heads, dim_ff=dim_ff)
output_layer = nn.Linear(d_model, vocab_size)  # project back to vocab size

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()   # better than MSE for classification
params = list(embedding.parameters()) + list(transformer.parameters()) + list(output_layer.parameters())
optimizer = optim.Adam(params, lr=0.001)

print("Training tiny transformer...")

for step in range(epochs):
    # Generate random toy data (batch=16, seq_len=10)
    x = torch.randint(0, vocab_size, (16, seq_len))   # integers in [0,vocab_size)
    x_embed = embedding(x)                            # (batch, seq_len, d_model)
    out = transformer(x_embed)                        # (batch, seq_len, d_model)
    logits = output_layer(out)                        # (batch, seq_len, vocab_size)

    # Loss: flatten predictions and targets
    loss = criterion(logits.view(-1, vocab_size), x.view(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

import torch
import torch.nn as nn

# ---- Multi-Head Attention ----
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, return_attn=False):
        B, L, D = q.shape

        # Project and reshape for heads
        q = self.q_linear(q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)  # (B, heads, L, L)
        out = attn @ v

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.fc(out)

        if return_attn:
            return out, attn
        return out


# ---- Transformer Block ----
class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dim_ff=128):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(x, x, x, return_attn=True)
            x = x + attn_out
            x = self.norm1(x)
            x = x + self.ff(x)
            x = self.norm2(x)
            return x, attn
        else:
            x = x + self.attn(x, x, x)
            x = self.norm1(x)
            x = x + self.ff(x)
            x = self.norm2(x)
            return x

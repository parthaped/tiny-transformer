
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention that supports:
      - returning attention matrices (return_attn=True)
      - per-head scaling/ablation via head_weights (tensor of shape [num_heads])
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, return_attn: bool = False, head_weights=None):
        """
        Args:
            q, k, v: (B, L, D)
            return_attn: if True, returns (out, attn) where attn is (B, H, L, L)
            head_weights: Optional tensor of shape (H,) with non-negative weights per head.
                          Use zeros to ablate heads; e.g., head_weights[i]=0 disables head i.
        """
        B, L, D = q.shape

        # Project and reshape for heads
        q = self.q_linear(q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.k_linear(k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        v = self.v_linear(v).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)
        head_out = attn @ v  # (B, H, L, d_k)

        # Optional per-head weights for ablation / replicator control
        if head_weights is not None:
            if head_weights.dim() == 1:
                hw = head_weights.view(1, -1, 1, 1)  # (1, H, 1, 1)
            else:
                # Allow broadcasting from (B,H) or (H,)
                # reshape best effort: (..., H) -> (B,H,1,1) if possible
                if head_weights.shape[-1] != self.num_heads:
                    raise ValueError("head_weights shape last dim must equal num_heads")
                while head_weights.dim() < 4:
                    head_weights = head_weights.unsqueeze(-1)
                hw = head_weights
            head_out = head_out * hw

        # Concatenate heads
        out = head_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.fc(out)

        if return_attn:
            return out, attn
        return out


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dim_ff=128, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn: bool = False, head_weights=None):
        if return_attn:
            attn_out, attn = self.attn(x, x, x, return_attn=True, head_weights=head_weights)
            x = x + attn_out
            x = self.norm1(x)
            x = x + self.ff(x)
            x = self.norm2(x)
            return x, attn
        else:
            x = x + self.attn(x, x, x, return_attn=False, head_weights=head_weights)
            x = self.norm1(x)
            x = x + self.ff(x)
            x = self.norm2(x)
            return x


import torch
import torch.nn as nn
from typing import Dict, List

@torch.no_grad()
def measure_head_payoffs(embedding: nn.Embedding,
                         block: nn.Module,
                         output_layer: nn.Module,
                         x_tokens: torch.Tensor,
                         criterion: nn.Module):
    """
    Measures per-head payoff as marginal loss reduction.
    payoff_i = base_loss - loss_with_head_i_ablated
    """
    device = next(block.parameters()).device
    x_tokens = x_tokens.to(device)
    x_embed = embedding(x_tokens)

    out = block(x_embed)
    logits = output_layer(out)
    base_loss = criterion(logits.view(-1, logits.size(-1)), x_tokens.view(-1))

    H = block.attn.num_heads
    losses = torch.zeros(H, device=device)
    for i in range(H):
        head_weights = torch.ones(H, device=device)
        head_weights[i] = 0.0
        out_i = block(x_embed, head_weights=head_weights)
        logits_i = output_layer(out_i)
        loss_i = criterion(logits_i.view(-1, logits_i.size(-1)), x_tokens.view(-1))
        losses[i] = loss_i

    payoffs = base_loss - losses
    return {
        'base_loss': base_loss.detach(),
        'payoffs': payoffs.detach(),
        'loss_with_ablation': losses.detach()
    }


@torch.no_grad()
def capture_attention_maps(block: nn.Module, x_embed: torch.Tensor):
    """
    Returns attention maps for a single forward pass without ablation.
    Args:
        block: TinyTransformerBlock
        x_embed: (B, L, D) embedded batch
    Returns:
        attn: (B, H, L, L)
    """
    x_out, attn = block(x_embed, return_attn=True)
    return attn

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (GPT-2 style)."""
    def __init__(self, n_features: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.cfg_model = cfg_model
        self.n_heads = cfg_model.n_heads
        self.head_dim = cfg_model.d_model // cfg_model.n_heads
        self.dropout = cfg_model.dropout
        
        self.qkv_proj = nn.Linear(cfg_model.d_model, 3*cfg_model.d_model, bias = cfg_model.bias)
        self.proj = nn.Linear(cfg_model.d_model, cfg_model.d_model, bias = cfg_model.bias)
        
        self.attn_drop = nn.Dropout(self.dropout)
        self.resid_drop = nn.Dropout(self.dropout)
        
        mask = torch.tril(torch.ones(cfg_model.block_size, cfg_model.block_size)).view(1, 1, cfg_model.block_size, cfg_model.block_size)
        self.register_buffer("causal_mask", mask, persistent=False)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim = 2)  # each [B, T, C]
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # [B, nh, T, hs]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        
        att = F.softmax(att, dim = -1)
        
        att = self.attn_drop(att)
        
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        y = self.resid_drop(self.proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.fc1 = nn.Linear(cfg_model.d_model, cfg_model.d_ff, bias = cfg_model.bias)
        self.fc2 = nn.Linear(cfg_model.d_ff, cfg_model.d_model, bias = cfg_model.bias)
        self.drop = nn.Dropout(cfg_model.dropout)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.ln1 = LayerNorm(cfg_model.d_model, bias = cfg_model.bias)
        self.ln2 = LayerNorm(cfg_model.d_model, bias = cfg_model.bias)
        self.attn = CausalSelfAttention(cfg_model)
        self.mlp = MLP(cfg_model)
        
    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.cfg_model = cfg_model
        self.tok_emb = nn.Embedding(cfg_model.vocab_size, cfg_model.d_model)
        self.pos_emb = nn.Embedding(cfg_model.block_size, cfg_model.d_model)
        self.drop = nn.Dropout(cfg_model.dropout)
        self.block = nn.ModuleList([Transformer(cfg_model) for _ in range(cfg_model.n_layers)])
        self.ln_f = LayerNorm(cfg_model.d_model, bias = cfg_model.bias)
        
        self.lm_head = nn.Linear(cfg_model.d_model, cfg_model.vocab_size, bias = False)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx : torch.Tensor,targets : Optional[torch.Tensor] = None, ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        
        for blk in self.block:
            x = blk(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx : torch.Tensor, max_new_tokens : int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, self.cfg_model.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:,-1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples =1)
            idx = torch.cat([idx, next_id], dim = 1)
        return idx
        
        
            
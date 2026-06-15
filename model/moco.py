"""MoCo for WSI Feature Contrastive Learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MoCoWSI(nn.Module):
    """MoCo v2 wrapper for WSI encoder with summary token pooling"""
    
    def __init__(self, base_encoder, embed_dim=512, proj_dim=128, K=16384, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        
        self.projector_q = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )
        self.projector_k = copy.deepcopy(self.projector_q)
        
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.K:
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :batch_size - (self.K - ptr)] = keys[self.K - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def _encode_and_pool(self, x, encoder, projector, summary_indices):
        """Encode and pool summary tokens"""
        encoded = encoder.encoder(x)
        summary_tokens = encoded[:, summary_indices, :]
        pooled = summary_tokens.mean(dim=1)
        projected = projector(pooled)
        return F.normalize(projected, dim=1)

    def forward(self, x_q, x_k, summary_indices):
        q = self._encode_and_pool(x_q, self.encoder_q, self.projector_q, summary_indices)
        
        with torch.no_grad():
            self._momentum_update()
            k = self._encode_and_pool(x_k, self.encoder_k, self.projector_k, summary_indices)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        self._dequeue_and_enqueue(k)
        
        return logits, labels


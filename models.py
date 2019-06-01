import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal=False):
        super(Transformer, self).__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h
    
class TransformerForLanguageModeling(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super(TransformerForLanguageModeling, self).__init__()
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings, num_max_positions, 
                                       num_heads, num_layers, dropout, causal=True)
        self.lm_head = nn.Linear(embed_dim, num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def forward(self, x, padding_mask=None):
        """ Input has shape [seq length, batch] """
        hidden_states = self.transformer(x, padding_mask)
        logits = self.lm_head(hidden_states)

        return logits
    
class TransformerForClassification(TransformerForLanguageModeling):
    def __init__(self, num_classes, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super(TransformerForClassification, self).__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, 
                                                           num_heads, num_layers, dropout)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, clf_tokens_mask, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        return clf_logits

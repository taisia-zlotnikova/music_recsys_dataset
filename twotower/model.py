"""MuQ + Qwen-название + теги; история → encoder → cross-attention декодера Qwen.

Ядро attention адаптировано из ../recsys/recsys/twotower/lazy.py.
Привязано к Qwen3 и transformers 4.57.6; не генерирует ответы, выдаёт retrieval-вектор.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_ITEM_FEATURES = ('audio', 'text', 'tags')


class TrackTower(nn.Module):
    def __init__(self, features, dim=256, channels=DEFAULT_ITEM_FEATURES):
        super().__init__()
        self.channels = tuple(channels)
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError('Нужен хотя бы один канал, без повторов')
        if 'tags' in self.channels:
            for key, dtype in [('tag_ids', torch.long), ('tag_weights', torch.float32)]:
                self.register_buffer(key, torch.tensor(features[key], dtype=dtype), persistent=False)
            self.tags = nn.EmbeddingBag(len(features['tag_vocab']) + 1, dim, mode='sum',
                                        padding_idx=len(features['tag_vocab']))
        for name in self.channels:
            if not name.isidentifier() or name.startswith('_'):
                raise ValueError(f'Некорректное имя признака: {name}')
            if name != 'tags':
                if name not in features or features[name].ndim != 2 or features[name].dtype.kind != 'f':
                    raise ValueError(f'Нет dense-признака {name!r} [число треков, размерность]')
                self.register_buffer('value_' + name, torch.tensor(features[name], dtype=torch.float32), persistent=False)
                # Имена audio_proj/text_proj сохранены для старых checkpoint.
                self.add_module(name + '_proj', nn.Sequential(nn.Linear(features[name].shape[1], dim), nn.LayerNorm(dim)))
        if 'tags' in self.channels:
            self.tag_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(len(self.channels) * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Linear(dim, dim)

    def forward(self):
        parts = []
        for name in self.channels:
            if name == 'tags':
                parts.append(self.tag_norm(self.tags(self.tag_ids, per_sample_weights=self.tag_weights)))
            else:
                parts.append(getattr(self, name + '_proj')(getattr(self, 'value_' + name)))
        r = self.mlp(torch.cat(parts, -1))
        return r, F.normalize(self.head(r), dim=-1)


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos.unsqueeze(1) + torch.cat([-x[..., half:], x[..., :half]], -1) * sin.unsqueeze(1)


def attend(q, k, v, mask, scale):
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return x.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def layer_step(layer, x, cos, sin, mask):
    att = layer.self_attn
    B, S, hd = x.shape[0], x.shape[1], att.head_dim
    h = layer.input_layernorm(x)
    q = rope(att.q_norm(att.q_proj(h).view(B, S, -1, hd)).transpose(1, 2), cos, sin)
    k = rope(att.k_norm(att.k_proj(h).view(B, S, -1, hd)).transpose(1, 2), cos, sin)
    v = att.v_proj(h).view(B, S, -1, hd).transpose(1, 2)
    x = x + att.o_proj(attend(q, k, v, mask, att.scaling))
    return x + layer.mlp(layer.post_attention_layernorm(x))


class CrossAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'):
            setattr(self, name, copy.deepcopy(getattr(layer.self_attn, name)).float())
        self.x_norm = copy.deepcopy(layer.input_layernorm).float()
        self.mem_norm = copy.deepcopy(layer.input_layernorm).float()
        self.head_dim, self.scale = layer.self_attn.head_dim, layer.self_attn.scaling
        self.gate = nn.Parameter(torch.zeros(()))
        self.requires_grad_(True)

    def forward(self, x, memory, mask):
        B, T, M, hd = x.shape[0], x.shape[1], memory.shape[1], self.head_dim
        m = self.mem_norm(memory)
        q = self.q_norm(self.q_proj(self.x_norm(x)).view(B, T, -1, hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(m).view(B, M, -1, hd)).transpose(1, 2)
        v = self.v_proj(m).view(B, M, -1, hd).transpose(1, 2)
        out = self.o_proj(attend(q, k.to(q.dtype), v.to(q.dtype), mask, self.scale))
        return x + (self.gate.tanh() * out).to(x.dtype)


def qwen_body(base):
    return base.get_base_model() if hasattr(base, 'get_base_model') else base


def positions(mask):
    return (mask.long().cumsum(1) - 1).clamp_min(0)


class UserTower(nn.Module):
    def __init__(self, base, dim=256, history_len=64, encoder_layers=2, cross_every=4, use_history=True):
        super().__init__()
        self.base = base.requires_grad_(False)
        body, hidden = qwen_body(base), base.config.hidden_size
        if base.config.model_type != 'qwen3':
            raise ValueError('Эта реализация рассчитана на Qwen3')
        if not 0 <= encoder_layers <= len(body.layers) or cross_every < 1:
            raise ValueError('Некорректное число encoder-слоёв или cross_every')
        self.checkpointing, self.use_history = True, use_history
        self.track_in = nn.Linear(dim, hidden)
        self.pos = nn.Embedding(history_len, hidden)
        self.count = nn.Embedding(16, hidden)
        self.age = nn.Embedding(16, hidden)
        for table in (self.pos, self.count, self.age):
            nn.init.normal_(table.weight, std=0.02)
        self.track_norm = nn.LayerNorm(hidden)
        # Масштаб токенов истории близок к масштабу обычных токенов Qwen.
        with torch.no_grad():
            norm = body.embed_tokens.weight[:min(20000, base.config.vocab_size)].float().norm(dim=1).mean().item()
            self.track_norm.weight.fill_(norm / math.sqrt(hidden))
        self.proj = nn.Linear(hidden, dim)
        self.encoder = nn.ModuleList(copy.deepcopy(layer).float().requires_grad_(True)
                                     for layer in list(body.layers)[:encoder_layers] if use_history)
        self.enc_norm = copy.deepcopy(body.norm).float().requires_grad_(True)
        self.cross = nn.ModuleDict({str(i): CrossAttention(body.layers[i])
                                   for i in range(0, len(body.layers), cross_every) if use_history})
        self.null = nn.Parameter(torch.randn(hidden) * norm / math.sqrt(hidden))

    def run(self, fn, *args):
        if self.training and self.checkpointing:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def memory(self, batch, reps):
        hm = batch['hmask']
        rank = (hm.flip(1).long().cumsum(1).flip(1) - 1).clamp(0, self.pos.num_embeddings - 1)
        mem = self.track_norm(self.track_in(reps[batch['hist']]) + self.pos(rank)
                              + self.count(batch['hcount']) + self.age(batch['hage']))
        if len(self.encoder) and hm.shape[1]:
            body = qwen_body(self.base)
            cos, sin = body.rotary_emb(mem, positions(hm))
            eye = torch.eye(hm.shape[1], dtype=torch.bool, device=hm.device)
            mask = (hm.unsqueeze(1) | eye).unsqueeze(1)
            for layer in self.encoder:
                mem = self.run(layer_step, layer, mem, cos, sin, mask)
            mem = self.enc_norm(mem)
        mem = torch.cat([self.null.to(mem.dtype).expand(mem.shape[0], 1, -1), mem], 1)
        mask = torch.cat([torch.ones(hm.shape[0], 1, device=hm.device, dtype=torch.bool), hm], 1)
        return mem, mask[:, None, None, :]

    def decoder_step(self, index, layer, x, cos, sin, mask, memory, memory_mask):
        if str(index) in self.cross:
            x = self.cross[str(index)](x, memory, memory_mask)
        return layer_step(layer, x, cos, sin, mask)

    def forward(self, batch, reps):
        body = qwen_body(self.base)
        x = body.embed_tokens(batch['tok'])
        memory, memory_mask = self.memory(batch, reps) if self.use_history else (None, None)
        valid, T = batch['mask'], x.shape[1]
        cos, sin = body.rotary_emb(x, positions(valid))
        eye = torch.eye(T, dtype=torch.bool, device=x.device)
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        mask = ((causal & valid.unsqueeze(1)) | eye).unsqueeze(1)
        for i, layer in enumerate(body.layers):
            x = self.run(self.decoder_step, i, layer, x, cos, sin, mask, memory, memory_mask)
        h = body.norm(x)[:, -1]  # У всех примеров последний токен — EOS, паддинг слева.
        with torch.autocast(h.device.type, enabled=False):
            return F.normalize(self.proj(h.float()), dim=-1)


def positive_loss(scores, targets):
    """Полный softmax; одинаковый вес каждого хода, равные веса его целей."""
    valid = targets >= 0
    logp = scores.float().log_softmax(-1).gather(1, targets.clamp_min(0))
    return -((logp * valid).sum(1) / valid.sum(1)).mean()


def retrieval_metrics(scores, targets, ks=(10, 100)):
    valid = targets >= 0
    n = valid.sum(1)
    top = scores.topk(min(max(ks), scores.shape[1]), dim=1).indices
    hits = ((top[:, :, None] == targets[:, None, :]) & valid[:, None, :]).any(-1)
    result = {}
    for k in ks:
        K = min(k, scores.shape[1])
        h = hits[:, :K]
        discount = 1 / torch.log2(torch.arange(K, device=scores.device).float() + 2)
        ideal = (torch.arange(K, device=scores.device)[None, :] < n[:, None]).float()
        result[f'recall@{k}'] = (h.sum(1) / n).sum().item()
        result[f'hit@{k}'] = h.any(1).float().sum().item()
        result[f'ndcg@{k}'] = ((h * discount).sum(1) / (ideal * discount).sum(1)).sum().item()
    return result


class TwoTower(nn.Module):
    def __init__(self, base, features, dim=256, item_features=DEFAULT_ITEM_FEATURES, **user_options):
        super().__init__()
        self.item = TrackTower(features, dim, item_features)
        self.user = UserTower(base, dim, **user_options)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.05)))

    def item_parts(self):
        with torch.autocast(next(self.item.parameters()).device.type, enabled=False):
            return self.item()

    def scores(self, batch, parts=None):
        reps, items = self.item_parts() if parts is None else parts
        users = self.user(batch, reps)
        with torch.autocast(users.device.type, enabled=False):
            return users.float() @ items.float().T / self.log_temperature.exp().clamp(0.01, 1.0)

    def forward(self, batch):
        return positive_loss(self.scores(batch), batch['targets'])


def build_model(features, config, device, local_files_only=False):
    from transformers import AutoModel
    from .data import model_path
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    base = AutoModel.from_pretrained(model_path(config['model'], local_files_only), dtype=dtype, attn_implementation='sdpa',
                                     local_files_only=local_files_only)
    model = TwoTower(base, features, dim=config['dim'], history_len=config['history_len'],
                     item_features=config.get('item_features', DEFAULT_ITEM_FEATURES),
                     encoder_layers=config['encoder_layers'], cross_every=config['cross_every'],
                     use_history=config['mode'] != 'text')
    if config.get('lora_r', 0):
        from peft import LoraConfig, get_peft_model
        model.user.base = get_peft_model(model.user.base, LoraConfig(
            r=config['lora_r'], lora_alpha=2 * config['lora_r'], lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    return model.to(device)


def trainable_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def load_trainable(model, state, allow_new_lora=False):
    own = model.state_dict()
    # PEFT добавляет base_model.model к пути. Это позволяет начать LoRA-этап из frozen-этапа.
    unexpected = set(state) - set(own)
    if unexpected:
        raise ValueError(f'Несовместимый checkpoint: {sorted(unexpected)[:3]}')
    required = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = required - set(state)
    if allow_new_lora:
        missing = {n for n in missing if 'lora_' not in n}
    if missing:
        raise ValueError(f'В checkpoint нет обучаемых весов: {sorted(missing)[:3]}')
    model.load_state_dict(state, strict=False)

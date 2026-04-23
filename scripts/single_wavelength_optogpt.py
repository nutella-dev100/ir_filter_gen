import copy, math, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# =========================================================
# TRANSFORMER CORE
# =========================================================

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x = attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, ff):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = ff
        self.sublayers = clones(SublayerConnection(size, 0.1), 3)

    def forward(self, x, memory, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory))
        return self.sublayers[2](x, self.ff)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.sublayers[0].norm.a_2.size(0))

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# =========================================================
# MODEL
# =========================================================

class SingleWaveOptoGPT(nn.Module):
    def __init__(self, encoder, decoder, embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.generator = generator

    def encode(self, x):
        return self.encoder(x).unsqueeze(1)

    def forward(self, src, tgt, tgt_mask):
        memory = self.encode(src)
        return self.decoder(self.embed(tgt), memory, tgt_mask)


class SpecEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------------------------
        # Continuous inputs:
        # [R, T, Dip, FOM]
        # -----------------------------------
        self.spec_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64,128),
            nn.ReLU()
        )

        # -----------------------------------
        # Substrate embedding
        # BK7=0, CaF2=1
        # -----------------------------------
        self.substrate_emb = nn.Embedding(2,16)

        # -----------------------------------
        # Fuse physics + substrate
        # -----------------------------------
        self.fc = nn.Sequential(
            nn.Linear(128+16,128),
            nn.ReLU()
        )


    def forward(self,x):
        """
        x = [R,T,Dip,FOM,substrate]
        shape: (batch,5)
        """

        # First four are continuous
        spec = x[:, :4]

        # Last column is substrate id
        substrate = x[:,4].long()

        spec_feat = self.spec_fc(spec)

        sub_feat = self.substrate_emb(substrate)

        combined = torch.cat(
            [spec_feat, sub_feat],
            dim=-1
        )

        return self.fc(combined)


def make_model(spec_dim, vocab):

    encoder = SpecEncoder()

    attn = MultiHeadedAttention(4, 128)
    ff = PositionwiseFeedForward(128, 256)

    decoder = Decoder(DecoderLayer(128, attn, attn, ff), 3)

    model = SingleWaveOptoGPT(
        encoder,
        decoder,
        nn.Sequential(Embeddings(128, vocab), PositionalEncoding(128)),
        Generator(128, vocab)
    )

    return model


# =========================================================
# VOCAB
# =========================================================

#MATERIALS = ['Al','Ag','SiO2','TiO2']
#THICKNESSES = list(range(1,102))
#SPECIAL = ['PAD','UNK','BOS','EOS']

def build_vocab_from_data(structures):
    SPECIAL = ['PAD','UNK','BOS','EOS']

    tokens = set()

    for struct in structures:
        for t in struct:
            if t not in SPECIAL:
                tokens.add(t)

    tokens = sorted(list(tokens))

    vocab = SPECIAL + tokens

    word2id = {w:i for i,w in enumerate(vocab)}
    id2word = {i:w for w,i in word2id.items()}

    return word2id, id2word


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(struct_path, spec_path):
    with open(struct_path, 'rb') as f:
        structures = pickle.load(f)
    with open(spec_path, 'rb') as f:
        spectra = pickle.load(f)
    return spectra, structures


# =========================================================
# BATCH
# =========================================================

def subsequent_mask(size):
    """
    Create a mask to hide future tokens.
    Output shape: (1, size, size)
    """
    mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    mask = torch.from_numpy(mask) == 0
    return mask.unsqueeze(0)


def collate_fn(batch, w2i, max_len=20):
    spectra, structures = zip(*batch)

    src = torch.tensor(np.array(spectra), dtype=torch.float32)

    tgt_ids = []
    for s in structures:
        ids = [w2i.get(t, w2i['UNK']) for t in s]
        ids = ids[:max_len] + [0]*(max_len-len(ids))
        tgt_ids.append(ids)

    tgt = torch.tensor(tgt_ids)

    tgt_in  = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    tgt_mask = subsequent_mask(tgt_in.size(1))

    return src, tgt_in, tgt_out, tgt_mask


def make_dataloader(spectra, structures, bs):
    data = list(zip(spectra, structures))
    random.shuffle(data)
    for i in range(0,len(data),bs):
        yield data[i:i+bs]

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, x, target):
        # x: (B*L, vocab)
        # target: (B*L,)
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))

            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[:, self.padding_idx] = 0

            mask = (target == self.padding_idx)
            true_dist[mask] = 0

        return self.criterion(x, true_dist)
    
def train_epoch(model, dataloader, optimizer, criterion, device, word2id):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        src, tgt_in, tgt_out, tgt_mask = collate_fn(batch, word2id)

        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        tgt_mask = tgt_mask.to(device)

        out = model(src, tgt_in, tgt_mask)
        log_probs = model.generator(out)

        loss = criterion(
            log_probs.reshape(-1, log_probs.size(-1)),
            tgt_out.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += tgt_out.numel()

    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device, word2id):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src, tgt_in, tgt_out, tgt_mask = collate_fn(batch, word2id)

            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            tgt_mask = tgt_mask.to(device)

            out = model(src, tgt_in, tgt_mask)
            log_probs = model.generator(out)

            loss = criterion(
                log_probs.reshape(-1, log_probs.size(-1)),
                tgt_out.reshape(-1)
            )

            total_loss += loss.item()
            total_tokens += tgt_out.numel()

    return total_loss / total_tokens

def greedy_decode(model, spec_target, word2id, id2word,
                  max_len=20, device='cpu'):

    model.eval()

    # --- Safety check ---
    assert len(spec_target) == 3, "Expected [R, T, substrate_id]"

    with torch.no_grad():
        src = torch.tensor(np.array([spec_target]), dtype=torch.float32).to(device)

        # Encode
        memory = model.encode(src)

        # Start with BOS
        ys = torch.tensor([[word2id['BOS']]], dtype=torch.long).to(device)

        generated = []

        for _ in range(max_len - 1):

            tgt_mask = subsequent_mask(ys.size(1)).to(device)

            out = model.decoder(
                model.embed(ys),
                memory,
                tgt_mask
            )

            log_probs = model.generator(out[:, -1])

            # --- Prevent PAD ---
            log_probs[:, word2id['PAD']] = -1e9

            # Greedy choice
            next_id = torch.argmax(log_probs, dim=-1).item()

            ys = torch.cat(
                [ys, torch.tensor([[next_id]], device=device)],
                dim=1
            )

            token = id2word[next_id]

            if token == 'EOS' and len(generated) > 0:
                break

            if token not in ['BOS', 'PAD', 'EOS']:
                generated.append(token)

    return generated
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scripts.single_wavelength_optogpt import (
    make_model, load_dataset, subsequent_mask
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
BEAM_SIZE  = 5

# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load("model.pt", map_location=DEVICE)
word2id = checkpoint["word2id"]
id2word = checkpoint["id2word"]

model = make_model(spec_dim=5, vocab=len(word2id)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

BOS = word2id['BOS']
EOS = word2id['EOS']
PAD = word2id['PAD']

# =====================
# LOAD VALIDATION DATA
# =====================
spec_data, struct_data = load_dataset(
    "processed/Structure_verification_set.pkl",
    "processed/Spectrum_verification_set.pkl"
)
print(f"Loaded {len(spec_data)} validation samples.")

# Build tensors once — avoids repeated per-sample conversions
spec_tensor = torch.tensor(np.array(spec_data), dtype=torch.float32)   # (N, 5)
val_dataset = TensorDataset(spec_tensor)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =====================
# GREEDY DECODE (batched)
# =====================
def greedy_decode_batch(model, src_batch: torch.Tensor,
                        max_len: int = 20) -> list[list[str]]:
    """
    Greedy decode for a batch of spectra.

    Args:
        src_batch : (B, 5) float tensor on the correct device.
        max_len   : maximum number of tokens to generate.

    Returns:
        List of B token lists, each excluding BOS/EOS/PAD.
    """
    B = src_batch.size(0)

    with torch.no_grad():
        memory = model.encode(src_batch)                        # (B, 1, 128)
        ys     = torch.full((B, 1), BOS, dtype=torch.long,
                            device=src_batch.device)            # (B, 1)
        done   = torch.zeros(B, dtype=torch.bool,
                             device=src_batch.device)
        results: list[list[str]] = [[] for _ in range(B)]

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).to(src_batch.device)
            out      = model.decoder(model.embed(ys), memory, tgt_mask)
            log_probs = model.generator(out[:, -1])

            # Prevent PAD
            log_probs[:, PAD] = -1e9

            # Prevent EOS at first step
            if ys.size(1) == 1:
                log_probs[:, EOS] = -1e9
            
            log_probs[done] = -1e9
            log_probs[done, EOS] = 0

            next_ids = torch.argmax(log_probs, dim=-1)          # (B,)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)

            for i, tok_id in enumerate(next_ids.tolist()):
                if done[i]:
                    continue
                token = id2word[tok_id]
                if token == 'EOS':
                    done[i] = True
                elif token not in ('BOS', 'PAD'):
                    results[i].append(token)

            if done.all():
                break

    return results


# =====================
# BEAM SEARCH DECODE (single sample)
# =====================
def beam_decode(model, spec_target, beam_size: int = 5,
                max_len: int = 20) -> list[tuple[float, list[str]]]:
    """
    Returns list of (score, token_list) sorted by score descending.
    Each candidate excludes BOS/EOS.
    """
    with torch.no_grad():
        src    = torch.tensor(np.array([spec_target]),
                              dtype=torch.float32).to(DEVICE)
        memory = model.encode(src)                              # (1, 1, 128)

        # beam: (cumulative_log_prob, ys_tensor, generated_tokens, is_done)
        beams = [(0.0,
                  torch.tensor([[BOS]], dtype=torch.long, device=DEVICE),
                  [],
                  False)]

        for _ in range(max_len - 1):
            new_beams = []

            for log_prob, ys, generated, done in beams:
                if done:
                    new_beams.append((log_prob, ys, generated, done))
                    continue

                tgt_mask      = subsequent_mask(ys.size(1)).to(DEVICE)
                out           = model.decoder(model.embed(ys), memory, tgt_mask)
                step_log_probs = model.generator(out[:, -1])   # (1, vocab)
                step_log_probs[:, PAD] = -1e9

                topk_probs, topk_ids = step_log_probs[0].topk(beam_size)

                for tok_lp, tok_id in zip(topk_probs.tolist(),
                                          topk_ids.tolist()):
                    token   = id2word[tok_id]
                    new_ys  = torch.cat(
                        [ys, torch.tensor([[tok_id]], device=DEVICE)], dim=1
                    )
                    new_gen = generated if token == 'EOS' else generated + [token]
                    new_beams.append(
                        (log_prob + tok_lp, new_ys, new_gen, token == 'EOS')
                    )

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

            if all(d for _, _, _, d in beams):
                break

    return sorted(
        [(s, gen) for s, _, gen, _ in beams],
        key=lambda x: x[0], reverse=True
    )


# =====================
# METRICS
# =====================
def _parse(tokens: list[str]) -> tuple[list[float], list[str]]:
    """['Ag_47', 'TiO2_4'] → ([47.0, 4.0], ['Ag', 'TiO2'])"""
    thicknesses, materials = [], []
    for t in tokens:
        if '_' in t:
            mat, th = t.rsplit('_', 1)
            try:
                thicknesses.append(float(th))
                materials.append(mat)
            except ValueError:
                pass
    return thicknesses, materials


def thickness_mae(pred_tokens, true_tokens) -> float:
    """
    Numeric MAE on layer thicknesses (nm).
    Missing layers are padded with 0 — a missing layer contributes
    its full expected thickness to the error.
    """
    p_th, _ = _parse(pred_tokens)
    t_th, _ = _parse(true_tokens)
    L = max(len(p_th), len(t_th), 1)
    p_th += [0.0] * (L - len(p_th))
    t_th += [0.0] * (L - len(t_th))
    return float(np.mean([abs(p - t) for p, t in zip(p_th, t_th)]))


def material_accuracy(pred_tokens, true_tokens) -> float:
    """Fraction of correctly predicted material types at each layer position."""
    _, p_mat = _parse(pred_tokens)
    _, t_mat = _parse(true_tokens)
    L = max(len(p_mat), len(t_mat))
    if L == 0:
        return 1.0
    p_mat += [''] * (L - len(p_mat))
    t_mat += [''] * (L - len(t_mat))
    return sum(p == t for p, t in zip(p_mat, t_mat)) / L


def layer_count_error(pred_tokens, true_tokens) -> int:
    """Absolute difference in number of predicted vs true layers."""
    p_th, _ = _parse(pred_tokens)
    t_th, _ = _parse(true_tokens)
    return abs(len(p_th) - len(t_th))


# =====================
# EVALUATE OVER FULL VALIDATION SET (batched)
# =====================
def get_true_tokens(idx: int) -> list[str]:
    return [t for t in struct_data[idx]
            if t not in ('BOS', 'EOS', 'PAD')]


print(f"\nRunning batched greedy decode (batch_size={BATCH_SIZE}) ...")

# --- Batched greedy over entire validation set ---
all_greedy: list[list[str]] = []
for (src_batch,) in val_loader:
    src_batch = src_batch.to(DEVICE)
    all_greedy.extend(greedy_decode_batch(model, src_batch))

print(f"Running beam search (beam_size={BEAM_SIZE}) per sample ...")

# Beam search is inherently sequential (variable-length tree expansion),
# so we iterate per sample — but greedy above already saves the bulk of
# the compute.
all_beam: list[list[str]] = []
for idx in range(len(spec_data)):
    beam_results = beam_decode(model, spec_data[idx], beam_size=BEAM_SIZE)
    all_beam.append(beam_results[0][1])

# --- Aggregate metrics ---
greedy_maes, beam_maes, mat_accs, layer_errs = [], [], [], []
for idx in range(len(spec_data)):
    true_tokens = get_true_tokens(idx)
    greedy_maes.append(thickness_mae(all_greedy[idx], true_tokens))
    beam_maes.append(thickness_mae(all_beam[idx],    true_tokens))
    mat_accs.append(material_accuracy(all_beam[idx], true_tokens))
    layer_errs.append(layer_count_error(all_beam[idx], true_tokens))

print(f"\n{'='*60}")
print(f"VALIDATION RESULTS  ({len(spec_data)} samples)")
print(f"{'='*60}")
print(f"Greedy decode  — Thickness MAE : {np.mean(greedy_maes):.4f} nm")
print(f"Beam search    — Thickness MAE : {np.mean(beam_maes):.4f} nm  "
      f"(beam_size={BEAM_SIZE})")
print(f"Beam search    — Material Acc. : {np.mean(mat_accs)*100:.2f}%")
print(f"Beam search    — Layer Count Err (mean abs): "
      f"{np.mean(layer_errs):.4f} layers")
print(f"{'='*60}")


# =====================
# SAMPLE PREDICTIONS (first 5)
# =====================
SUBSTRATE_INV = {0: "BK7", 1: "CaF2"}

print(f"\nSample predictions (first 5 validation samples):")
print(f"{'-'*60}")

for idx in range(min(5, len(spec_data))):
    spec        = spec_data[idx]
    true_tokens = get_true_tokens(idx)

    greedy_pred  = all_greedy[idx]  # already computed via greedy_decode_batch
    beam_results = [(None, all_beam[idx])]

    R         = spec[0]
    T         = spec[1]
    Dip       = spec[2] * 10 + 60
    FOM       = spec[3] * 300
    substrate = SUBSTRATE_INV[int(round(spec[4]))]

    print(f"\n[Sample {idx}]")
    print(f"  Inputs: R={R:.4f}, T={T:.4f}, "
          f"Dip={Dip:.2f}, FOM={FOM:.2f}, Substrate={substrate}")
    print(f"  True structure   : {true_tokens}")
    print(f"  Greedy decode    : {greedy_pred}  "
          f"(MAE={thickness_mae(greedy_pred, true_tokens):.2f} nm)")
    print(f"  Beam search top-3:")
    for rank, (score, tokens) in enumerate(beam_results[:3]):
        mae = thickness_mae(tokens, true_tokens)
        print(f"    [{rank+1}] score={score:.3f}  MAE={mae:.2f} nm  {tokens}")
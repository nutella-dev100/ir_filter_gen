import torch
import numpy as np
from scripts.single_wavelength_optogpt import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load("model.pt", map_location=DEVICE)
word2id = checkpoint["word2id"]
id2word = checkpoint["id2word"]

model = make_model(spec_dim=3, vocab=len(word2id)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

# =====================
# LOAD VALIDATION DATA
# =====================
spec_data, struct_data = load_dataset(
    "dataset/Structure_verification_set.pkl",
    "dataset/Spectrum_verification_set.pkl"
)

print(f"Loaded {len(spec_data)} validation samples.")


# =====================
# BEAM SEARCH DECODE
# =====================
def beam_decode(model, spec_target, word2id, id2word,
                beam_size=5, max_len=20, device='cpu'):
    """
    Returns list of (score, token_list) sorted by score descending.
    Each candidate excludes BOS/EOS.
    """
    model.eval()
    with torch.no_grad():
        src = torch.tensor(np.array([spec_target]), dtype=torch.float32).to(device)
        memory = model.encode(src)  # (1, 1, 128)

        BOS = word2id['BOS']
        EOS = word2id['EOS']

        # beam: (cumulative_log_prob, ys_tensor, generated_tokens, is_done)
        beams = [(0.0, torch.tensor([[BOS]], dtype=torch.long, device=device), [], False)]

        for _ in range(max_len - 1):
            new_beams = []

            for log_prob, ys, generated, done in beams:
                if done:
                    new_beams.append((log_prob, ys, generated, done))
                    continue

                tgt_mask = subsequent_mask(ys.size(1)).to(device)
                out = model.decoder(model.embed(ys), memory, tgt_mask)
                step_log_probs = model.generator(out[:, -1])  # (1, vocab)

                topk_probs, topk_ids = step_log_probs[0].topk(beam_size)

                for tok_lp, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                    token = id2word[tok_id]
                    new_ys = torch.cat(
                        [ys, torch.tensor([[tok_id]], device=device)], dim=1
                    )
                    new_gen = generated if token == 'EOS' else generated + [token]
                    new_beams.append((log_prob + tok_lp, new_ys, new_gen, token == 'EOS'))

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

            if all(done for _, _, _, done in beams):
                break

        results = sorted([(s, gen) for s, _, gen, _ in beams], key=lambda x: x[0], reverse=True)
        return results


# =====================
# METRICS
# =====================
def _parse(tokens):
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


def thickness_mae(pred_tokens, true_tokens):
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


def material_accuracy(pred_tokens, true_tokens):
    """Fraction of correctly predicted material types at each layer position."""
    _, p_mat = _parse(pred_tokens)
    _, t_mat = _parse(true_tokens)

    L = max(len(p_mat), len(t_mat))
    if L == 0:
        return 1.0

    p_mat += [''] * (L - len(p_mat))
    t_mat += [''] * (L - len(t_mat))

    return sum(p == t for p, t in zip(p_mat, t_mat)) / L


def layer_count_error(pred_tokens, true_tokens):
    """Absolute difference in number of predicted vs true layers."""
    p_th, _ = _parse(pred_tokens)
    t_th, _ = _parse(true_tokens)
    return abs(len(p_th) - len(t_th))


# =====================
# EVALUATE OVER FULL VALIDATION SET
# =====================
BEAM_SIZE = 5

greedy_maes, beam_maes = [], []
mat_accs, layer_errs = [], []

print(f"\nRunning greedy + beam search (beam_size={BEAM_SIZE}) over all samples...")

for idx in range(len(spec_data)):
    spec = spec_data[idx]
    true_tokens = [t for t in struct_data[idx] if t not in ('BOS', 'EOS', 'PAD')]

    greedy_pred = greedy_decode(model, spec, word2id, id2word, device=DEVICE)
    beam_results = beam_decode(model, spec, word2id, id2word,
                               beam_size=BEAM_SIZE, device=DEVICE)
    best_beam = beam_results[0][1]

    greedy_maes.append(thickness_mae(greedy_pred, true_tokens))
    beam_maes.append(thickness_mae(best_beam, true_tokens))
    mat_accs.append(material_accuracy(best_beam, true_tokens))
    layer_errs.append(layer_count_error(best_beam, true_tokens))

print(f"\n{'='*60}")
print(f"VALIDATION RESULTS  ({len(spec_data)} samples)")
print(f"{'='*60}")
print(f"Greedy decode  — Thickness MAE : {np.mean(greedy_maes):.4f} nm")
print(f"Beam search    — Thickness MAE : {np.mean(beam_maes):.4f} nm  (beam_size={BEAM_SIZE})")
print(f"Beam search    — Material Acc. : {np.mean(mat_accs)*100:.2f}%")
print(f"Beam search    — Layer Count Err (mean abs): {np.mean(layer_errs):.4f} layers")
print(f"{'='*60}")

# =====================
# SAMPLE PREDICTIONS (first 5)
# =====================
print(f"\nSample predictions (first 5 validation samples):")
print(f"{'-'*60}")

for idx in range(min(5, len(spec_data))):
    spec = spec_data[idx]
    true_tokens = [t for t in struct_data[idx] if t not in ('BOS', 'EOS', 'PAD')]

    greedy_pred = greedy_decode(model, spec, word2id, id2word, device=DEVICE)
    beam_results = beam_decode(model, spec, word2id, id2word,
                               beam_size=BEAM_SIZE, device=DEVICE)

    print(f"\n[Sample {idx}]")
    print(f"  Spectrum (R, T, substrate)  : {spec}")
    print(f"  True structure   : {true_tokens}")
    print(f"  Greedy decode    : {greedy_pred}  "
          f"(MAE={thickness_mae(greedy_pred, true_tokens):.2f} nm)")
    print(f"  Beam search top-3:")
    for rank, (score, tokens) in enumerate(beam_results[:3]):
        mae = thickness_mae(tokens, true_tokens)
        print(f"    [{rank+1}] score={score:.3f}  MAE={mae:.2f} nm  {tokens}")

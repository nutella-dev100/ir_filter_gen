import torch
import torch.optim as optim
import os

from scripts.single_wavelength_optogpt import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# CONFIG
# =====================
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-4

TRAIN_SPEC = "processed/Spectrum_train_set.pkl"
TRAIN_STRUC = "processed/Structure_train_set.pkl"

VAL_SPEC = "processed/Spectrum_verification_set.pkl"
VAL_STRUC = "processed/Structure_verification_set.pkl"

# =====================
# LOAD DATA
# =====================
print("Loading dataset...")

train_spec, train_struct = load_dataset(TRAIN_STRUC, TRAIN_SPEC)
val_spec, val_struct = load_dataset(VAL_STRUC, VAL_SPEC)

print(f"Train: {len(train_spec)} samples")
print(f"Val:   {len(val_spec)} samples")

# =====================
# VOCAB
# =====================
word2id, id2word = build_vocab_from_data(train_struct)

# =====================
# MODEL
# =====================
model = make_model(spec_dim=5, vocab=len(word2id)).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = LabelSmoothingLoss(len(word2id))

# =====================
# TRAIN LOOP
# =====================
for epoch in range(EPOCHS):

    train_loader = make_dataloader(train_spec, train_struct, BATCH_SIZE)
    val_loader   = make_dataloader(val_spec, val_struct, BATCH_SIZE)

    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, word2id)
    val_loss   = evaluate(model, val_loader, criterion, DEVICE, word2id)

    print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

# =====================
# SAVE
# =====================
os.makedirs("models", exist_ok=True)

torch.save({
    "model": model.state_dict(),
    "word2id": word2id,
    "id2word": id2word
}, "models/model.pt")

print("Model saved.")
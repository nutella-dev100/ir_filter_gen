import pickle
from collections import Counter

with open("processed/Structure_train_set.pkl", "rb") as f:
    structures = pickle.load(f)

print("Sample structures:\n")

for i in range(10):
    print(structures[i])

print("\nChecking for bad tokens...\n")

bad_tokens = set()

for struct in structures:
    for t in struct:
        if t in ['BOS', 'EOS', 'PAD']:
            continue

        parts = t.split('_')
        if len(parts) != 2 or not parts[1].isdigit() or parts[0] == "":
            bad_tokens.add(t)

print("Bad tokens found:", bad_tokens)

print("\nMaterial vocabulary:\n")

materials = set()

for struct in structures:
    for t in struct:
        if t not in ['BOS', 'EOS', 'PAD']:
            materials.add(t.split('_')[0])

print(materials)

counter = Counter()

for s in structures:
    for t in s:
        if t.startswith(("FG_", "MoS2_", "Anti_")):
            counter[t] += 1

print(counter)
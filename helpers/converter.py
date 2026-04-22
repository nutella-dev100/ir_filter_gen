import pandas as pd
import numpy as np
import os
import pickle as pkl
import ast

# =========================
# CONFIG
# =========================
source_dir = 'data'
files_to_convert = ['train_set.csv', 'verification_set.csv']
output_dir = 'processed'

# Change this if needed
R_COL = 'R_1550nm'
T_COL = 'T_1550nm'

# Substrate mapping based on requirements
substrate_map = {"BK7": 0, "CaF2": 1}

os.makedirs(output_dir, exist_ok=True)

VALID_MATERIALS = {
    'Ag', 'Au', 'Al',
    'SiO2', 'TiO2', 'Si',
    'MoS2', 'FG', 'Anti'
}


# =========================
# HELPER
# =========================
def process_structure(s):
    try:
        tokens = ast.literal_eval(s)

        if not isinstance(tokens, list):
            return None

        cleaned = []
        for t in tokens:
            if not isinstance(t, str):
                return None

            # STRICT check: must be material_thickness
            parts = t.split('_')

            if len(parts) != 2:
                return None

            material, thickness = parts

            # reject invalid materials
            if material not in VALID_MATERIALS:
                print(f"Rejected token: {t}")
                return None

            if not thickness.isdigit():
                return None

            # Material must be non-empty string
            if len(material) == 0:
                return None

            # Thickness must be integer
            if not thickness.isdigit():
                return None

            cleaned.append(f"{material}_{int(thickness)}")

        return ['BOS'] + cleaned + ['EOS']

    except:
        return None


# =========================
# MAIN LOOP
# =========================
for file_name in files_to_convert:

    path = os.path.join(source_dir, file_name)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue

    print(f"\nProcessing {file_name}...")

    df = pd.read_csv(path)

    spectra = []
    structures = []

    for idx, row in df.iterrows():

        # --- Spectrum ---
        try:
            R = float(row[R_COL])
            T = float(row[T_COL])
            
            # Extract substrate and map it to an integer
            substrate = row['substrate']
            sub_val = substrate_map[substrate]
            
        except (KeyError, ValueError, TypeError):
            # Skip rows with missing columns or substrates not in the map
            continue 

        # Updated to include the substrate index
        spec = [R, T, sub_val]

        # --- Structure ---
        struct = process_structure(row['structure'])
        if struct is None:
            continue

        spectra.append(spec)
        structures.append(struct)

    spectra = np.array(spectra, dtype=np.float32)

    print(f"Valid samples: {len(spectra)}")

    # --- Save ---
    set_name = file_name.replace('.csv', '')

    structure_file = os.path.join(output_dir, f'Structure_{set_name}.pkl')
    spectrum_file  = os.path.join(output_dir, f'Spectrum_{set_name}.pkl')

    with open(structure_file, 'wb') as f:
        pkl.dump(structures, f)

    with open(spectrum_file, 'wb') as f:
        pkl.dump(spectra, f)

    print(f"Saved:")
    print(f"  {structure_file}")
    print(f"  {spectrum_file}")
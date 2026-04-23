import pandas as pd
import numpy as np
import os
import pickle as pkl
import ast

# =========================
# CONFIG
# =========================
source_dir = 'data'
files_to_convert = [
    'train_set.csv',
    'verification_set.csv'
]

output_dir = 'processed'

R_COL = 'R_1550nm'
T_COL = 'T_1550nm'

# substrate encoding
substrate_map = {
    "BK7": 0,
    "CaF2": 1
}

os.makedirs(output_dir, exist_ok=True)

# valid structure tokens
VALID_MATERIALS = {
    'Ag', 'Au', 'Al',
    'SiO2', 'TiO2', 'Si',
    'MoS2', 'FG', 'Anti'
}

# =========================
# TOKEN VALIDATION
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

            parts = t.split('_')

            # must be material_thickness
            if len(parts) != 2:
                return None

            material, thickness = parts

            # reject bad material tokens
            if material not in VALID_MATERIALS:
                print(f"Rejected token: {t}")
                return None

            # thickness must be integer string
            if not thickness.isdigit():
                return None

            cleaned.append(
                f"{material}_{int(thickness)}"
            )

        # add BOS/EOS
        return ['BOS'] + cleaned + ['EOS']

    except:
        return None


# =========================
# MAIN
# =========================
for file_name in files_to_convert:

    path = os.path.join(
        source_dir,
        file_name
    )

    if not os.path.exists(path):
        print(f"Warning: {path} missing")
        continue

    print(f"\nProcessing {file_name}...")

    df = pd.read_csv(path)

    spectra = []
    structures = []

    for idx, row in df.iterrows():

        # -------------------
        # LABELS
        # -------------------
        try:

            R = float(row[R_COL])
            T = float(row[T_COL])

            # substrate encoding
            substrate = row['substrate']
            sub_val = substrate_map[substrate]

            # normalized labels
            dip = (float(row["Dip"]) - 60)/10
            fom = float(row["FOM"])/300

        except (
            KeyError,
            ValueError,
            TypeError
        ):
            continue

        # final target vector
        # [R,T,Dip,FOM,Substrate]
        spec = [
            R,
            T,
            dip,
            fom,
            sub_val
        ]

        # -------------------
        # STRUCTURE
        # -------------------
        struct = process_structure(
            row["structure"]
        )

        if struct is None:
            continue

        spectra.append(spec)
        structures.append(struct)

    spectra = np.array(
        spectra,
        dtype=np.float32
    )

    print(
        f"Valid samples: {len(spectra)}"
    )

    # -------------------
    # SAVE
    # -------------------
    set_name = file_name.replace(
        '.csv',
        ''
    )

    structure_file = os.path.join(
        output_dir,
        f"Structure_{set_name}.pkl"
    )

    spectrum_file = os.path.join(
        output_dir,
        f"Spectrum_{set_name}.pkl"
    )

    with open(
        structure_file,
        'wb'
    ) as f:
        pkl.dump(
            structures,
            f
        )

    with open(
        spectrum_file,
        'wb'
    ) as f:
        pkl.dump(
            spectra,
            f
        )

    print("Saved:")
    print(structure_file)
    print(spectrum_file)
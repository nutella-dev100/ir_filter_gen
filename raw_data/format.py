import pandas as pd
import os

# =========================
# CONFIG
# =========================
FOLDER_PATH = r"D:\Acads\SOP\novel\raw_data"
OUTPUT_FILE = "combined_1550nm_data_clean.csv"

all_data = []

# =========================
# HELPERS
# =========================
def to_nm(thickness_m):
    try:
        return max(1, int(round(float(thickness_m)*1e9)))
    except:
        return None


def extract_materials(filename):
    """
    Example:
    BK7_Ag_25_55_SiO2_1_10_FG_1_10.xlsx

    returns:
    substrate, metal, dielectric, 2D
    """
    parts = filename.replace(".xlsx","").split("_")

    substrate = parts[0] if len(parts) > 0 else None
    metal = parts[1] if len(parts) > 1 else None
    dielectric = parts[4] if len(parts) > 4 else None
    material_2d = parts[7] if len(parts) > 7 else None

    return substrate, metal, dielectric, material_2d


def build_structure(row, metal, dielectric, material_2d, df_cols):

    structure = []

    # Metal always exists
    if 'tMt' in df_cols and pd.notna(row['tMt']):
        nm = to_nm(row['tMt'])
        if nm:
            structure.append(f"{metal}_{nm}")

    # 3-layer case (has tDi)
    if 'tDi' in df_cols:

        if pd.notna(row['tDi']):
            nm = to_nm(row['tDi'])
            if nm:
                structure.append(f"{dielectric}_{nm}")

        if material_2d and 't2d' in df_cols and pd.notna(row['t2d']):
            nm = to_nm(row['t2d'])
            if nm:
                structure.append(f"{material_2d}_{nm}")

    # 2-layer case (no tDi)
    else:
        if dielectric and 't2d' in df_cols and pd.notna(row['t2d']):
            nm = to_nm(row['t2d'])
            if nm:
                structure.append(f"{dielectric}_{nm}")

    return structure


# =========================
# PROCESS FILES
# =========================
for file in os.listdir(FOLDER_PATH):

    if not file.endswith(".xlsx"):
        continue

    file_path = os.path.join(FOLDER_PATH, file)
    print(f"Processing {file}")

    try:
        substrate, metal_name, dielectric_name, material_2d_name = extract_materials(file)

        df = pd.read_excel(file_path)
        cols = df.columns

        if "Rmin" not in cols or "tMt" not in cols:
            print(f"Skipping {file}")
            continue

        # -------------------
        # Labels
        # -------------------
        df["R_1550nm"] = df["Rmin"]
        df["T_1550nm"] = 1 - df["Rmin"]

        df["FOM"] = df["FOM"]
        df["Dip"] = df["Dip"]

        df["substrate"] = substrate

        # -------------------
        # Structure
        # -------------------
        df["structure"] = df.apply(
            lambda row:
                build_structure(
                    row,
                    metal_name,
                    dielectric_name,
                    material_2d_name,
                    cols
                ),
            axis=1
        )

        # remove malformed rows
        df = df[df["structure"].apply(lambda x: len(x) >= 2)]

        df["num_layers"] = df["structure"].apply(len)

        # -------------------
        # Final export columns
        # -------------------
        df_final = df[
            [
                "structure",
                "num_layers",
                "substrate",
                "R_1550nm",
                "T_1550nm",
                "FOM",
                "Dip"
            ]
        ]

        all_data.append(df_final)

    except Exception as e:
        print(f"Error in {file}: {e}")


# =========================
# SAVE
# =========================
if len(all_data)==0:
    print("No valid data.")
else:
    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDone.")
    print("Samples:", len(combined_df))
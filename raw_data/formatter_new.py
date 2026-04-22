import pandas as pd
import os

# =========================
# CONFIG
# =========================
FOLDER_PATH = r"D:\Acads\SOP\novel\data"
OUTPUT_FILE = "combined_1550nm_data_clean.csv"

all_data = []

# =========================
# HELPERS
# =========================
def to_nm(thickness_m):
    """Convert meters → nm"""
    try:
        return max(1, int(round(float(thickness_m) * 1e9)))
    except:
        return None


def extract_materials(filename):
    """
    Extract materials from filename
    Works for both 2-layer and 3-layer
    """
    parts = filename.replace(".xlsx", "").split("_")

    metal = parts[1] if len(parts) > 1 else None
    dielectric = parts[4] if len(parts) > 4 else None
    material_2d = parts[7] if len(parts) > 7 else None

    return metal, dielectric, material_2d


def build_structure(row, metal, dielectric, material_2d, df_cols):
    structure = []

    # --- Metal (always present) ---
    if 'tMt' in df_cols and pd.notna(row['tMt']):
        nm = to_nm(row['tMt'])
        if nm:
            structure.append(f"{metal}_{nm}")

    # --- CASE 1: 3-layer (tDi exists) ---
    if 'tDi' in df_cols:
        # Dielectric
        if pd.notna(row['tDi']):
            nm = to_nm(row['tDi'])
            if nm:
                structure.append(f"{dielectric}_{nm}")

        # 2D material
        if material_2d and 't2d' in df_cols and pd.notna(row['t2d']):
            nm = to_nm(row['t2d'])
            if nm:
                structure.append(f"{material_2d}_{nm}")

    # --- CASE 2: 2-layer (NO tDi → t2d is dielectric) ---
    else:
        if 't2d' in df_cols and pd.notna(row['t2d']):
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
    print(f"Processing: {file}")

    try:
        # --- Extract materials ---
        metal_name, dielectric_name, material_2d_name = extract_materials(file)

        df = pd.read_excel(file_path)
        df_cols = df.columns

        # --- Validate ---
        if "Rmin" not in df_cols or "tMt" not in df_cols:
            print(f"⚠️ Skipping {file} (missing required columns)")
            continue

        # --- R and T ---
        df["R_1550nm"] = df["Rmin"]
        df["T_1550nm"] = 1 - df["Rmin"]

        # --- Build structure ---
        df["structure"] = df.apply(
            lambda row: build_structure(
                row,
                metal_name,
                dielectric_name,
                material_2d_name,
                df_cols
            ),
            axis=1
        )

        # --- Remove invalid rows ---
        df = df[df["structure"].apply(lambda x: len(x) >= 2)]

        # --- Num layers ---
        df["num_layers"] = df["structure"].apply(len)

        # --- Final selection ---
        df_final = df[[
            "structure",
            "num_layers",
            "R_1550nm",
            "T_1550nm"
        ]]

        all_data.append(df_final)

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")


# =========================
# COMBINE + SAVE
# =========================
if len(all_data) == 0:
    print("❌ No valid data found.")
else:
    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Done. Saved as {OUTPUT_FILE}")
    print(f"Total samples: {len(combined_df)}")
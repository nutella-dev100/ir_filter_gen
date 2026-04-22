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
    return max(1, int(round(thickness_m * 1e9)))

def extract_materials(filename):
    """
    Example:
    BK7_Ag_25_55_SiO2_1_10_FG_1_10.xlsx
    → Ag, SiO2, FG
    """
    parts = filename.replace(".xlsx", "").split("_")

    metal = parts[1]
    dielectric = parts[4]
    material_2d = parts[7]

    return metal, dielectric, material_2d

# =========================
# PROCESS FILES
# =========================
for file in os.listdir(FOLDER_PATH):
    if file.endswith(".xlsx"):
        file_path = os.path.join(FOLDER_PATH, file)
        print(f"Processing: {file}")

        # --- Extract material names from filename ---
        metal_name, dielectric_name, material_2d_name = extract_materials(file)

        df = pd.read_excel(file_path)

        # --- R and T ---
        df["R_1550nm"] = df["Rmin"]
        df["T_1550nm"] = 1 - df["Rmin"]

        # --- Build structure ---
        def build_structure(row):
            return [
                f"{metal_name}_{to_nm(row['tMt'])}",
                f"{dielectric_name}_{to_nm(row['tDi'])}",
                f"{material_2d_name}_{to_nm(row['t2d'])}"
            ]

        df["structure"] = df.apply(build_structure, axis=1)

        df["num_layers"] = 3

        df_final = df[[
            "structure",
            "num_layers",
            "R_1550nm",
            "T_1550nm"
        ]]

        all_data.append(df_final)

# =========================
# COMBINE + SAVE
# =========================
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone. Saved as {OUTPUT_FILE}")
print(f"Total samples: {len(combined_df)}")
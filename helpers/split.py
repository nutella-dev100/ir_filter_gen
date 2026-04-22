import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your dataset
# Replace 'your_data.csv' with the actual filename
input_file = 'raw_data/combined_1550nm_data_clean.csv'
df = pd.read_csv(input_file)

# 2. Split the data
# train_size=0.8 sets the training set to 80%
# random_state ensures reproducibility (same split every time you run it)
train_df, verify_df = train_test_split(df, train_size=0.8, random_state=42)

# 3. Save to two separate CSV files
train_df.to_csv('train_set.csv', index=False)
verify_df.to_csv('verification_set.csv', index=False)

print(f"Split complete!")
print(f"Training set: {len(train_df)} rows")
print(f"Verification set: {len(verify_df)} rows")
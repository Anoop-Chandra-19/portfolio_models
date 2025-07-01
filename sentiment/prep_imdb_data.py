import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "sentiment/data/IMDB Dataset.csv"
OUT_DIR = "sentiment/data"
TRAIN_OUT = os.path.join(OUT_DIR, "train.csv")
VAL_OUT = os.path.join(OUT_DIR, "val.csv")
TEST_OUT = os.path.join(OUT_DIR, "test.csv")

# Load and shuffle data
df = pd.read_csv(DATA_PATH)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df[["review", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (80%), temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Split temp into validation (10%) and test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# Save to CSV
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Files saved as:\n- {TRAIN_OUT}\n- {VAL_OUT}\n- {TEST_OUT}")


from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------
# 1. Define project paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    BASE_DIR / "dataset" / "raw" /
    "PS_20174392719_1491204439457_log.csv"
)

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Load original dataset
# -----------------------------------

print("Loading dataset...")

df = pd.read_csv(RAW_DATA_PATH)

print("Original dataset shape:", df.shape)

# -----------------------------------
# 3. Select required features
# -----------------------------------

selected_columns = [
    "step",
    "type",
    "amount",
    "isFraud"
]

df = df[selected_columns].copy()

print("Selected dataset shape:", df.shape)

# -----------------------------------
# 4. Separate features and target
# -----------------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# -----------------------------------
# 5. Encode transaction type
# -----------------------------------

X = pd.get_dummies(
    X,
    columns=["type"],
    dtype=int
)

print("\nEncoded features:")
print(X.columns.tolist())

# -----------------------------------
# 6. Split dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining and testing split completed!")

# -----------------------------------
# 7. Save processed datasets
# -----------------------------------

X_train.to_csv(
    PROCESSED_DIR / "X_train.csv.gz",
    index=False,
    compression="gzip"
)

X_test.to_csv(
    PROCESSED_DIR / "X_test.csv.gz",
    index=False,
    compression="gzip"
)

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv.gz",
    index=False,
    compression="gzip"
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv.gz",
    index=False,
    compression="gzip"
)

# -----------------------------------
# 8. Display final results
# -----------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nFraud percentage in training data:")
print(round(y_train.mean() * 100, 4), "%")

print("\nFraud percentage in testing data:")
print(round(y_test.mean() * 100, 4), "%")

print("\nProcessed files saved successfully!")

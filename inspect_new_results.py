import pandas as pd

print("=== 8B Results ===")
df8 = pd.read_csv("evaluated_results_200_8b.csv")
print(df8.shape)
print(df8.columns.tolist())
print("\nSample columns preview:")
print(df8.head(2).T)

print("\n=== 70B Results ===")
df70 = pd.read_csv("evaluated_results_200_70b.csv")
print(df70.shape)
print(df70.columns.tolist())
print("\nSample columns preview:")
print(df70.head(2).T)

# Check common columns and possible score columns
common_cols = list(set(df8.columns) & set(df70.columns))
print(f"\nCommon columns: {common_cols}")
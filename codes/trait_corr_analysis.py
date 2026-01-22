import pandas as pd

# Load the CSV file
df = pd.read_csv("../Trait-Correlation/trait_corr.csv")  # Replace with actual file path

# Ensure only the 5 relevant columns are selected (in correct order)
ocean_columns = ["O", "C", "E", "A", "N"]
ocean_df = df[ocean_columns]

# Compute pairwise Pearson correlation matrix
correlation_matrix = ocean_df.corr(method='pearson')

# Display correlation matrix
print("Pairwise Pearson Correlation Matrix (OCEAN traits):")
print(correlation_matrix.round(2))

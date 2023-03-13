import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({'Class': ['A', 'B', 'C', 'B', 'A', 'C', 'C']})

# Replace all values in the 'Class' column that are not 'A' with 0, and all 'A' values with 1
df['Class'] = df['Class'].replace({'A': 1, 'B': 0, 'C': 0})

# Print the updated DataFrame
print(df)

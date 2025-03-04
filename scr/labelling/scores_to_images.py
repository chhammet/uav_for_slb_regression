import pandas as pd

# File paths
scores = "DJI_202307270907_032-035_G3-2023.csv"
image_to_row = "G3_fid_to_row/G3_merged_output.csv"
output_file = "0727_G3_scored_to_image.csv"

# Load CSV files
scores_df = pd.read_csv(scores)
image_to_rows_df = pd.read_csv(image_to_row)

# Ensure 'row' column is of the same type in both DataFrames
scores_df['row'] = scores_df['row'].astype(str)  # Convert to string
image_to_rows_df['row'] = image_to_rows_df['row'].astype(str)  # Convert to string


# Merge on the 'row' column, keeping all columns from DJI dataset
scored_to_image_df = scores_df.merge(image_to_rows_df[['row', 'image']], on='row', how='left')

# Save the merged dataset
scored_to_image_df.to_csv(output_file, index=False)

print(f"Merged file saved as: {output_file}")
# Define the parent directory containing subdirectories with CSV pairs
parent_directory = "G3_fid_to_row"

# Output file path
output_csv_path = os.path.join(parent_directory, "G3_merged_output.csv")

# Prepare list to store extracted data
data = []

# Iterate over all subdirectories in the parent directory
for subdirectory in os.listdir(parent_directory):
    subdirectory_path = os.path.join(parent_directory, subdirectory)
    
    if os.path.isdir(subdirectory_path):
        csv_files = [f for f in os.listdir(subdirectory_path) if f.endswith(".csv")]
        
        # Ensure there are exactly two CSV files in the subdirectory
        if len(csv_files) == 2:
            csv_1_path = os.path.join(subdirectory_path, csv_files[0])
            csv_2_path = os.path.join(subdirectory_path, csv_files[1])
            
            # Read the CSV files
            df1 = pd.read_csv(csv_1_path, header=None, dtype=str)  # Read as string to preserve formatting
            df2 = pd.read_csv(csv_2_path, header=None, dtype=str)
            
            # Ensure both CSVs have the same shape
            if df1.shape == df2.shape:
                num_rows, num_cols = df1.shape
                
                # Determine the max length of elements to pad accordingly
                max_length = max(df1.astype(str).map(len).max().max(), 
                                 df2.astype(str).map(len).max().max())
                
                for row in range(num_rows):
                    for col in range(num_cols):
                        csv1_element = str(df1.iloc[row, col]).zfill(max_length)  # Pad with leading zeros
                        csv2_element = str(df2.iloc[row, col]).zfill(max_length)
                        
                        data.append([
                            csv1_element,  # Element from first CSV with leading zeros
                            csv2_element,  # Element from second CSV with leading zeros
                            row,           # Row index
                            col,           # Column index
                            subdirectory   # Directory name
                        ])
            else:
                print(f"Skipping {subdirectory} due to mismatched CSV shapes.")
        else:
            print(f"Skipping {subdirectory} due to incorrect number of CSV files.")

# Create DataFrame and save to CSV
output_df = pd.DataFrame(data, columns=["image", "row", "Row_Index", "Column_Index", "Directory"])
output_df.to_csv(output_csv_path, index=False)

print(f"Output CSV saved to: {output_csv_path}")
import pickle
import pandas as pd
import os

def main():
    """
    Reads data from a pickle file and saves it to a CSV file.
    """
    # Using the path found in the 'cwq' directory.
    input_pkl_path = '/root/SubgraphRAG/retrieve/data_files/cwq/processed/tsmc_test.pkl'
    
    # Define the output path in the 'retrieve' directory for easy access.
    output_csv_path = '/root/SubgraphRAG/retrieve/tsmc_cwq_test_data.csv'

    print(f"Attempting to load data from: {input_pkl_path}")

    if not os.path.exists(input_pkl_path):
        print(f"Error: Input file not found at {input_pkl_path}")
        print("Please check the file path.")
        return

    try:
        # Load data from the pickle file
        with open(input_pkl_path, 'rb') as f:
            data = pickle.load(f)

        print("Data loaded successfully.")

        # Check if the data is in a format that can be easily converted to a DataFrame
        if not isinstance(data, (list, dict, pd.DataFrame)):
            print(f"Warning: The data in the pickle file is of type '{type(data)}', which may not convert cleanly to a CSV.")
            print("Attempting conversion anyway.")

        # It's common for these pickle files to contain a list of dictionaries.
        # Pandas handles this structure perfectly.
        df = pd.DataFrame(data)

        # Save DataFrame to CSV
        # Using utf-8-sig to ensure compatibility with Excel for files with BOM
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        print(f"Successfully converted and saved data to: {output_csv_path}")
        print(f"The CSV file contains {len(df)} rows and {len(df.columns)} columns.")
        print("Columns:", df.columns.tolist())

    except Exception as e:
        print(f"An error occurred during the conversion process: {e}")

if __name__ == '__main__':
    main()

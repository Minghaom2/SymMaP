import json
import csv
import argparse
import os 

def main(input_dir, output_file):
    x_file_path = os.path.join(input_dir, "X.json")
    y_file_path = os.path.join(input_dir, "y.json")

    try:
        # Load the data
        with open(x_file_path, "r", encoding="utf-8") as f:
            x = json.load(f)

        with open(y_file_path, "r", encoding="utf-8") as f:
            y = json.load(f)

        with open(output_file, mode='w', newline='', encoding='utf-8') as file:  # Use 'w' mode and specify encoding
            writer = csv.writer(file)
            for i in range(len(x)):
                row = x[i] + [y[i]]
                writer.writerow(row)
        
        print("Data saved to", output_file)

    except Exception as e:
        print("Error processing files:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON data to CSV.")
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing X.json and y.json')
    parser.add_argument('--output-file', type=str, required=True, help='Output CSV file')

    args = parser.parse_args()
    main(args.input_dir, args.output_file)


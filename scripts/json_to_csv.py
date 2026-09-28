import argparse
import csv
import json


def main():
    arg_parser = argparse.ArgumentParser(description="Convert JSON to CSV")
    arg_parser.add_argument(
        "input", type=str, help="Path to the input JSON file"
    )
    arg_parser.add_argument(
        "output", type=str, help="Path to the output CSV file"
    )
    args = arg_parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output

    if not input_file_path.endswith(".json"):
        raise ValueError("Input file must be a JSON file.")

    if not output_file_path.endswith(".csv"):
        raise ValueError("Output file must be a CSV file.")

    with open(input_file_path, encoding="utf-8") as file:
        data = json.load(file)

    with open(output_file_path, "w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(dict.fromkeys(key for row in data for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully converted {input_file_path} to {output_file_path}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime
import os

def normalize_version(version):
    if pd.isna(version):
        return version
    v = str(version).lower().strip()
    if v.startswith("v"):
        v = v[1:]
    return v

def get_major_minor_version(version):
    if pd.isna(version):
        return version
    parts = str(version).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version

def add_version_prefix(version):
    if pd.isna(version):
        return version
    v = str(version)
    return v if v.startswith("v") else f"v{v}"

def streamline_from_dataframe_dynamic(df, merge_versions=True, merge_minor_versions=True):
    """
    Dynamically streamline a dataframe based on column name patterns.

    Args:
        df (pandas.DataFrame): Input dataframe
        merge_versions (bool): Whether to normalize and merge similar versions
        merge_minor_versions (bool): Whether to merge minor versions (e.g., v1.17.x -> v1.17)

    Returns:
        pandas.DataFrame: The streamlined dataframe
    """

    working_df = df.copy()

    # Normalize versions if requested
    if merge_versions:
        working_df.iloc[:, 0] = working_df.iloc[:, 0].apply(normalize_version)
        if merge_minor_versions:
            working_df.iloc[:, 0] = working_df.iloc[:, 0].apply(get_major_minor_version)

    cols = working_df.columns.tolist()
    version_col = working_df.iloc[:, 0]

    streamlined_data = {"Version": version_col}

    # Define keyword-based column groups
    groups = {
        "Done": ["done"],
        "In QA": ["in qa", "qa"],
        "In Review": ["in review", "review"],
        "To Do": ["to do", "todo"],
        "Internal Test": ["internal test"],
        "In Progress": ["in progress"]
    }

    # Assign columns dynamically
    for label, keywords in groups.items():
        matching_cols = [c for c in cols if any(k in c.lower() for k in keywords)]
        if matching_cols:
            streamlined_data[label] = working_df[matching_cols].sum(axis=1)

    streamlined_df = pd.DataFrame(streamlined_data)

    # If merging versions, group by version
    if merge_versions:
        numeric_columns = [c for c in streamlined_df.columns if c != "Version"]
        streamlined_df = streamlined_df.groupby("Version", as_index=False).agg({c: "sum" for c in numeric_columns})

    # Reorder columns
    desired_order = ["Version", "In QA", "In Review", "Internal Test", "To Do", "In Progress", "Done"]
    final_columns = [c for c in desired_order if c in streamlined_df.columns]
    final_df = streamlined_df[final_columns]

    # Add version prefix
    final_df["Version"] = final_df["Version"].apply(add_version_prefix)

    return final_df

def create_sample_data():
    sample_data = {
        'Version': ['1.16', '1.17', '1.17.2', '1.17.5', '1.18', '1.19', '1.2'],
        'IN QA': [0, 0, 0, 0, 0, 0, 0],
        'IN REVIEW': [0, 0, 0, 0, 0, 0, 0],
        'INTERNAL TEST': [0, 0, 0, 0, 0, 0, 0],
        'In QA': [0, 0, 0, 0, 0, 0, 0],
        'In Review': [0, 0, 0, 0, 0, 0, 0],
        'To Do': [1, 0, 0, 0, 0, 0, 5],
        'In Progress': [0, 0, 0, 0, 0, 0, 0],
        'Done': [2, 0, 1, 0, 5, 1, 8],
        'To Do.1': [1, 0, 0, 0, 8, 0, 0],
        'Done.1': [1, 0, 0, 4, 6, 0, 4],
        'To Do.2': [0, 0, 0, 0, 0, 0, 0],
        'In Progress.1': [0, 0, 0, 0, 0, 0, 0],
        'Done.2': [2, 2, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(sample_data)

def main():
    parser = argparse.ArgumentParser(
        description='Streamline table data by consolidating columns according to name patterns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py input.csv
  python script.py input.csv -o custom_output.csv
  python script.py input.csv --no-merge-minor
  python script.py input.csv --no-merge-versions
"""
    )

    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (default: auto-generated)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-merge-versions', action='store_true', help='Disable version merging')
    parser.add_argument('--no-merge-minor', action='store_true', help='Disable merging minor versions (e.g. 1.17.2 stays separate)')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    if not args.input_file.lower().endswith('.csv'):
        print(f"Error: Input file must be a CSV file. Got: {args.input_file}")
        sys.exit(1)

    try:
        if args.verbose:
            print(f"Reading input file: {args.input_file}")

        df = pd.read_csv(args.input_file)

        if args.verbose:
            print(f"Input data shape: {df.shape}")
            print("Original columns:")
            for i, col in enumerate(df.columns):
                print(f"  {i + 1}: {col}")

        merge_versions = not args.no_merge_versions
        merge_minor_versions = not args.no_merge_minor

        result_df = streamline_from_dataframe_dynamic(df, merge_versions=merge_versions, merge_minor_versions=merge_minor_versions)

        output_file = args.output or f"{os.path.splitext(os.path.basename(args.input_file))[0]}_streamlined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        result_df.to_csv(output_file, index=False)

        print(f"✅ Successfully processed data!")
        print(f"📁 Input file: {args.input_file}")
        print(f"💾 Output file: {output_file}")
        print(f"📊 Output shape: {result_df.shape}")

        if args.verbose:
            print("\nStreamlined columns:")
            for i, col in enumerate(result_df.columns):
                print(f"  {i + 1}: {col}")
            print("\nFirst 5 rows:")
            print(result_df.head().to_string())

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def show_sample_usage():
    print("Table Streamlining Script - Sample Usage")
    print("=" * 50)

    sample_df = create_sample_data()
    print("Sample input data:")
    print(sample_df.head())

    print("\nStreamlined result:")
    result = streamline_from_dataframe_dynamic(sample_df)
    print(result.head())

    print("\nCommand line usage:")
    print("  python script.py input.csv")
    print("  python script.py input.csv -o output.csv")
    print("  python script.py input.csv --no-merge-minor")
    print("  python script.py input.csv --no-merge-versions")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_sample_usage()
    else:
        main()
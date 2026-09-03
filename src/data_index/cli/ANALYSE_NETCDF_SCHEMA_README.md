# NetCDF Schema Variance Analysis

This script analyzes the structure and variance of NetCDF datasets stored in the S3 Tables Iceberg catalog. It helps identify schema inconsistencies across collections and facilities, which is crucial for understanding data quality and compatibility.

## Overview

NetCDF datasets often have variant schemas, especially when:
- Datasets span multiple decades of collection
- Different facilities or projects manage data independently
- Codebase changes or hands-offs occur over time

This script examines the `variables` column across records to quantify schema variance and identify which variable schemas are most common.

## Features

- **Typer CLI Framework**: Professional command-line interface with auto-generated help
- **Rich Output**: Colored, formatted output with tables and formatting
- **AWS Credential Handling**: Automatic credential detection and setup
- **Schema Normalization**: Treats schemas with identical variables (in any order) as equivalent
- **Variable Analysis**: Identifies unique and missing variables per schema
- **Flexible Output**: Save results to custom directories
- **Performance**: Column pruning and row filtering for efficient S3 queries

## Prerequisites

- Python 3.10+
- `uv` package manager
- `boto3` (for AWS credential handling)
- Access to AWS S3 Tables with appropriate credentials
- AWS credentials configured via one of:
  - AWS CLI: `aws configure`
  - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - IAM role (if running on EC2 or ECS)

## Installation

1. Install with CLI dependencies:
   ```bash
   uv sync --group cli
   ```

2. Configure AWS credentials using one of these methods:

   **Option A: AWS CLI Configuration**
   ```bash
   aws configure
   # Enter your AWS Access Key ID and Secret Access Key when prompted
   ```

   **Option B: Environment Variables**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

   **Option C: IAM Role**
   If running on EC2 or ECS, the script will automatically use your IAM role credentials.

3. Verify credentials are working:
   ```bash
   aws sts get-caller-identity
   ```

## Usage

### CLI Commands

View help:
```bash
uv run python analyse_netcdf_schema.py --help
```

Analyse all data:
```bash
uv run python analyse_netcdf_schema.py
```

Analyse a specific S3 prefix:
```bash
uv run python analyse_netcdf_schema.py "IMOS/ANFOG/%"
```

Analyse with custom output directory:
```bash
uv run python analyse_netcdf_schema.py "IMOS/SOOP/SOOP-CO2%" --output-dir ./results
```

Other examples:
```bash
uv run python analyse_netcdf_schema.py "IMOS/SOOP/%"
uv run python analyse_netcdf_schema.py "IMOS/ANMN%"
```

### CLI Options

- `S3_PREFIX` (optional): S3 prefix to analyse (e.g., `'IMOS/ANFOG/%'`, `'IMOS/SOOP/SOOP-CO2%'`)
- `--output-dir, -o`: Directory to save results (default: current directory)
- `--help`: Show help message

The script will display results to console with a rich formatted table and save detailed analysis to a timestamped output file:
- `netcdf_schema_analysis_all_20260716_141000.txt` (for all data)
- `netcdf_schema_analysis_IMOS_ANFOG_wildcard_20260716_141000.txt` (for IMOS/ANFOG/%)

### Programmatic Usage

```python
from analyse_netcdf_schema import load_table, query_with_filter

# Load the table
table = load_table()

# Query with S3 prefix filter
df = query_with_filter(
    table, s3_prefix="IMOS/ANFOG/%", selected_fields=("key", "variables")
)

# Analyze variance
print(df["variables"].value_counts(sort=True))
```

## Functions

### `load_table(namespace, table_name, region, arn)`

Loads an Iceberg table from the S3 Tables catalog with automatic AWS credential handling.

**Parameters:**
- `namespace` (str): Iceberg namespace, default: `"data_index"`
- `table_name` (str): Table name, default: `"structured_metadata_v5"`
- `region` (str): AWS region, default: `"ap-southeast-2"`
- `arn` (str): S3 Tables bucket ARN

**Returns:** `pyiceberg.Table` object

### `query_with_filter(table, s3_prefix, row_filter, selected_fields)`

Queries the table with filtering and column pruning for performance.

**Parameters:**
- `table`: Iceberg table object
- `s3_prefix` (str, optional): S3 prefix to filter by (e.g., `"IMOS/ANFOG/%"`)
- `row_filter` (str, optional): Additional row filter predicate
- `selected_fields` (tuple, optional): Fields to retrieve (column pruning)

**Returns:** Polars DataFrame

### `analyze_schema_variance(df, group_by)`

Analyzes schema variance by examining the `variables` column with normalization.

**Key Features:**
- Schemas with the same variables in any order are treated as identical
- Tracks which variables are unique to each schema
- Identifies missing variables in each schema variant
- Identifies directory prefixes associated with each schema

**Parameters:**
- `df`: Polars DataFrame with `variables` and `key` columns
- `group_by` (str, optional): Column to group analysis by

**Returns:** Dict with normalized schema variants, variable-level analysis, and their associated directory prefixes

### `save_results(results, s3_prefix)`

Saves analysis results to a timestamped output file.

**Parameters:**
- `results`: Analysis results dictionary from `analyze_schema_variance()`
- `s3_prefix` (str, optional): S3 prefix used for querying (used in filename)

**Returns:** Output file path

## Performance Tips

1. **Use field pruning** - Select only the fields you need:
   ```python
   selected_fields = ("key", "variables")
   ```

2. **Apply row filters** - Filter at scan time to reduce data transfer:
   ```python
   row_filter = "key like 'IMOS/ANFOG/%'"
   ```

3. **Use Polars** - The script returns Polars DataFrames for efficient in-memory operations

## Output Format

Results are saved to a timestamped text file. Example output:

```
================================================================================
NetCDF Schema Variance Analysis
================================================================================

Timestamp: 2026-07-16T14:19:10.123456
Query Prefix: IMOS/SOOP/SOOP-CO2%
Total Records: 731
Total Distinct Schemas: 2
Total Unique Variables: 22

================================================================================
All Variables Found
================================================================================
  - AIRT_raw
  - ATMP_uncorr_raw
  - CO2_STD_Value
  - ...
  - xH2O_PPM_raw

================================================================================
Schema Variance Details
================================================================================

Schema #1
  Variables (20): ['AIRT_raw', 'ATMP_uncorr_raw', 'CO2_STD_Value', ...]
  Record Count: 450
  Variables Unique to This Schema:
    + Diff_Press_Equ_raw
    + LabMain_sw_flow_raw
  Variables Missing from This Schema:
    - xCO2_ADJUSTED
  Associated Prefixes (2):
    - IMOS/SOOP/SOOP-CO2/VLMJ_Investigator/REALTIME/2022/11/
    - IMOS/SOOP/SOOP-CO2/VLMJ_Investigator/REALTIME/2023/01/

Schema #2
  Variables (22): ['AIRT_raw', 'ATMP_uncorr_raw', 'CO2_STD_Value', ...]
  Record Count: 281
  Variables Unique to This Schema:
    + xCO2_ADJUSTED
  Variables Missing from This Schema:
  Associated Prefixes (1):
    - IMOS/SOOP/SOOP-CO2/VLMJ_Investigator/HISTORICAL/
```

**Key features:**
- Schemas with identical variables in any order are now treated as the same schema
- "All Variables Found" section lists every variable across all schemas
- Each schema shows:
  - Variable count and list
  - Record count (number of files)
  - Variables unique to this schema (marked with `+`)
  - Variables missing from this schema that exist elsewhere (marked with `-`)
  - Associated directory prefixes where this schema appears
- Helps identify which datasets have incomplete variable sets

## Output Interpretation

The `value_counts()` output shows:
- **Index**: Variable schema (as a list or string)
- **Count**: Number of records with that schema

Example:
```
variable_schema                                count
["temp", "sal", "depth"]                       1250
["temp", "sal", "depth", "cond"]               450
["temp", "sal"]                                200
```

High variance (many unique schemas) indicates inconsistent data collection or processing.

## Examples

### Analyse a specific facility

```python
df = query_with_filter(
    table, s3_prefix="IMOS/ANFOG/%", selected_fields=("key", "facility", "variables")
)
print(df["variables"].value_counts(sort=True).head(10))
```

Or from the command line:
```bash
uv run python analyse_netcdf_schema.py "IMOS/ANFOG/%"
```

### Find records with specific variables

```python
df = query_with_filter(table, selected_fields=("key", "variables"))
df_with_temp = df.filter(pl.col("variables").str.contains("temp"))
print(f"Records with 'temp' variable: {len(df_with_temp)}")
```

### Group analysis by facility

```python
df = query_with_filter(table, selected_fields=("key", "facility", "variables"))
grouped = df.group_by("facility").agg(
    pl.col("variables").value_counts().alias("schemas")
)
print(grouped)
```

## Troubleshooting

**Connection Issues:**
- Verify AWS credentials: `aws sts get-caller-identity`
- Check S3 Tables ARN is correct
- Ensure IAM role has S3 Tables access permissions

**"Unable to locate AWS credentials" error:**
- Verify you've configured credentials using one of the methods above
- Check environment variables: `echo $AWS_ACCESS_KEY_ID`
- Try: `aws configure` to set up credentials

**Empty Results:**
- Verify row filter syntax (uses SQL WHERE clause syntax)
- Check table name and namespace
- Ensure the S3 prefix contains data (try without prefix first)

**Performance Issues:**
- Reduce selected fields
- Add stricter row filters
- Process in smaller batches if needed

## Output Data Types

The script returns Polars DataFrames where:
- `key` (str): S3 object key
- `size` (int): Object size in bytes
- `variables` (list): List of variable names in the NetCDF file
- `facility` (str): IMOS facility name
- `collection` (str): Legacy collection name

## Related Resources

- [Iceberg Documentation](https://iceberg.apache.org/)
- [Polars Documentation](https://docs.pola-rs.com/)
- [NetCDF Documentation](https://www.unidata.ucar.edu/software/netcdf/)

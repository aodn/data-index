#!/usr/bin/env python3
"""
NetCDF Schema Variance Analysis CLI Tool

Analyzes the structure and variance of NetCDF datasets stored in S3 Tables Iceberg catalog.
Identifies schema inconsistencies across collections and facilities.
"""

from data_index.iceberg_config import S3TablesCatalogConfig, IcebergTableConfig
import polars as pl
import boto3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Analyze NetCDF schema variance in S3 Tables")
console = Console()


def load_table(
    namespace: str = "data_index",
    table_name: str = "structured_metadata_v5",
    region: str = "ap-southeast-2",
    arn: str = "arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
):
    """Load an Iceberg table from S3 Tables catalog."""
    # Setup AWS credentials from the current environment or IAM role
    session = boto3.Session()
    credentials = session.get_credentials()
    
    if credentials is None:
        raise RuntimeError(
            "Unable to locate AWS credentials. Please configure credentials via:\n"
            "  - AWS CLI: aws configure\n"
            "  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            "  - IAM role (if running on EC2 or ECS)"
        )
    
    frozen = credentials.get_frozen_credentials()
    
    # Set environment variables for pyiceberg
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    
    catalog_config = S3TablesCatalogConfig(region=region, arn=arn)
    table_config = IcebergTableConfig(
        catalog_config=catalog_config,
        namespace=namespace,
        table_name=table_name,
    )
    return table_config.load()


def query_with_filter(
    table,
    s3_prefix: str = None,
    row_filter: str = None,
    selected_fields: tuple = None,
):
    """
    Query table with optional S3 prefix filter and column pruning.
    
    Args:
        table: Iceberg table object
        s3_prefix: S3 prefix to filter by (e.g., 'IMOS/ANFOG/%')
        row_filter: Additional row filter predicate
        selected_fields: Fields to retrieve (column pruning)
    
    Returns:
        Polars DataFrame
    """
    # Build combined row filter
    filters = []
    if s3_prefix:
        filters.append(f"key like '{s3_prefix}'")
    if row_filter:
        filters.append(f"({row_filter})")
    
    combined_filter = " AND ".join(filters) if filters else None
    
    scan = table.scan(
        row_filter=combined_filter,
        selected_fields=selected_fields,
    )
    return scan.to_polars()


def analyze_schema_variance(df: pl.DataFrame, group_by: str = None):
    """
    Analyze schema variance by examining variables column.
    Schemas with the same variables (in any order) are considered equivalent.

    Args:
        df: Polars DataFrame with variables column and key column
        group_by: Optional column to group analysis by (e.g., 'facility', 'collection')

    Returns:
        Dict with variance analysis results including prefix associations
    """
    results = {}

    # Normalize schemas by sorting variables for comparison
    # Create a new column with sorted variables as a string for grouping
    def normalize_schema(variables):
        if isinstance(variables, list):
            return str(sorted(variables))
        elif isinstance(variables, str):
            return str(sorted(eval(variables)))
        else:
            return str(sorted(variables))
    
    df_normalized = df.with_columns(
        pl.col("variables")
        .map_elements(normalize_schema, return_dtype=pl.Utf8)
        .alias("normalized_schema")
    )
    
    # Count records per normalized schema
    schema_counts = df_normalized["normalized_schema"].value_counts(sort=True)
    
    # For each normalized schema, find the associated prefixes and collect variables
    schema_prefixes = []
    all_variables = set()
    
    for schema_row in schema_counts.rows(named=True):
        normalized_schema = schema_row["normalized_schema"]
        count = schema_row["count"]
        
        # Find records with this normalized schema
        records_with_schema = df_normalized.filter(pl.col("normalized_schema") == normalized_schema)
        
        # Extract directory path from keys by removing the filename
        records_with_schema = records_with_schema.with_columns(
            pl.col("key")
            .str.replace(r'/[^/]*$', '/')
            .alias("dir_prefix")
        )
        
        # Get unique prefixes for this schema, sorted
        prefixes = records_with_schema["dir_prefix"].unique().sort().to_list()
        
        # Get actual variables from the original data (unsorted)
        sample_vars = records_with_schema["variables"].to_list()[0]
        if isinstance(sample_vars, str):
            sample_vars = eval(sample_vars)
        
        # Collect all variables
        all_variables.update(sample_vars)
        
        schema_prefixes.append({
            "schema": str(sorted(sample_vars)),
            "schema_display": str(sample_vars),
            "sorted_variables": sorted(sample_vars),
            "count": count,
            "prefixes": prefixes,
            "num_prefixes": len(prefixes)
        })
    
    # Analyze which variables are common/unique to each schema
    variable_analysis = []
    for idx, schema_info in enumerate(schema_prefixes):
        schema_vars = set(schema_info["sorted_variables"])
        
        # Find variables unique to this schema
        other_vars = set().union(*[set(s["sorted_variables"]) for i, s in enumerate(schema_prefixes) if i != idx]) if len(schema_prefixes) > 1 else set()
        unique_vars = schema_vars - other_vars
        
        # Find variables missing from this schema
        missing_vars = all_variables - schema_vars
        
        schema_info["unique_variables"] = sorted(unique_vars)
        schema_info["missing_variables"] = sorted(missing_vars)
        
        variable_analysis.append(schema_info)
    
    results["schema_variance"] = variable_analysis
    results["total_schemas"] = len(variable_analysis)
    results["total_records"] = len(df)
    results["all_variables"] = sorted(all_variables)

    return results


def save_results(results: dict, s3_prefix: str = None):
    """
    Save analysis results to a timestamped output file.
    
    Args:
        results: Analysis results dictionary
        s3_prefix: S3 prefix that was queried (used in filename)
    
    Returns:
        Path to the output file
    """
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_part = s3_prefix.replace("/", "_").replace("%", "wildcard") if s3_prefix else "all"
    output_file = f"netcdf_schema_analysis_{prefix_part}_{timestamp}.txt"
    
    # Write results
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NetCDF Schema Variance Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Query Prefix: {s3_prefix or 'All data'}\n")
        f.write(f"Total Records: {results['total_records']}\n")
        f.write(f"Total Distinct Schemas: {results['total_schemas']}\n")
        f.write(f"Total Unique Variables: {len(results['all_variables'])}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("All Variables Found\n")
        f.write("=" * 80 + "\n")
        for var in results["all_variables"]:
            f.write(f"  - {var}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Schema Variance Details\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, schema_info in enumerate(results["schema_variance"], 1):
            f.write(f"Schema #{idx}\n")
            f.write(f"  Variables ({len(schema_info['sorted_variables'])}): {schema_info['schema_display']}\n")
            f.write(f"  Record Count: {schema_info['count']}\n")
            
            if schema_info["unique_variables"]:
                f.write(f"  Variables Unique to This Schema:\n")
                for var in schema_info["unique_variables"]:
                    f.write(f"    + {var}\n")
            
            if schema_info["missing_variables"]:
                f.write(f"  Variables Missing from This Schema:\n")
                for var in schema_info["missing_variables"]:
                    f.write(f"    - {var}\n")
            
            f.write(f"  Associated Prefixes ({schema_info['num_prefixes']}):\n")
            for prefix in schema_info["prefixes"]:
                f.write(f"    - {prefix}\n")
            f.write("\n")
    
    return output_file


@app.command()
def analyse(
    s3_prefix: Optional[str] = typer.Argument(
        None,
        help="S3 prefix to analyse (e.g., 'IMOS/ANFOG/%', 'IMOS/SOOP/SOOP-CO2%')"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (default: current directory)"
    )
):
    """
    Analyze NetCDF schema variance in S3 Tables.
    
    Examines the structure of NetCDF datasets and identifies schema inconsistencies
    across collections and facilities.
    """
    try:
        console.print("[bold blue]Loading S3 Table...[/bold blue]")
        table = load_table()
        
        console.print("\n[bold]=== Querying Data ===[/bold]")
        df_all = query_with_filter(
            table,
            s3_prefix=s3_prefix,
            selected_fields=("key", "variables"),
        )
        console.print(f"Total records: [green]{len(df_all)}[/green]")
        if s3_prefix:
            console.print(f"Prefix: [cyan]{s3_prefix}[/cyan]")
        
        console.print("\n[bold]=== Analyzing Schema Variance ===[/bold]")
        results = analyze_schema_variance(df_all)
        
        console.print(f"Total distinct schemas: [yellow]{results['total_schemas']}[/yellow]")
        console.print(f"Total unique variables: [yellow]{len(results['all_variables'])}[/yellow]")
        
        # Display top 10 schemas in a table
        console.print(f"\n[bold]Top 10 schemas by record count:[/bold]")
        table_display = Table(title="Schema Summary")
        table_display.add_column("#", style="dim")
        table_display.add_column("Records", justify="right")
        table_display.add_column("Prefixes", justify="right")
        table_display.add_column("Variables", style="cyan")
        
        for idx, schema_info in enumerate(results["schema_variance"][:10], 1):
            table_display.add_row(
                str(idx),
                str(schema_info['count']),
                str(schema_info['num_prefixes']),
                schema_info['schema'][:60] + "..." if len(schema_info['schema']) > 60 else schema_info['schema']
            )
        console.print(table_display)
        
        console.print("\n[bold]=== Saving Results ===[/bold]")
        # Change to output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            original_cwd = os.getcwd()
            os.chdir(output_dir)
            try:
                output_file = save_results(results, s3_prefix)
            finally:
                os.chdir(original_cwd)
            output_path = output_dir / output_file
        else:
            output_file = save_results(results, s3_prefix)
            output_path = Path(output_file)
        
        console.print(f"Results saved to: [green]{output_path.absolute()}[/green]")
        
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

## Deleting Tables
https://docs.aws.amazon.com/cli/latest/reference/s3tables/delete-table.html

```bash
aws s3tables delete-table \
--region ap-southeast-2 \
--table-bucket-arn <value> \
--namespace <value> \
--name <value> 
```

## Deleting Table Buckets

```bash
aws s3tables delete-table-bucket \
--region ap-southeast-2 \
--table-bucket-arn <value>
```

## IAM Permissions

The default cluster flow reads from the live inventory S3 Table and writes to two data-index S3 Tables (with flow-run-specific suffixes during cluster runs).


### Scopes
| Access scope | Table bucket ARN | Namespace / table(s) | Required IAM actions |
| --- | --- | --- | --- |
| Read source inventory | `arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3` | `b_imos-data.inventory` | `s3tables:GetTable`, `s3tables:GetTableMetadataLocation`, `s3tables:GetTableData` |
| Provision + write structured/unstructured outputs | `arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index` | Namespace: `data_index`; Tables: `structured_metadata_v*`, `unstructured_metadata*` (including flow-run suffix variants) | `s3tables:CreateNamespace`, `s3tables:CreateTable`, `s3tables:GetTable`, `s3tables:GetTableMetadataLocation`, `s3tables:UpdateTableMetadataLocation`, `s3tables:PutTableData`, `s3tables:GetTableData` |

### Why these actions are needed

- `LiveS3InventorySource` reads the inventory Iceberg table via S3 Tables catalog APIs.
- `StructuredS3TableSink` and `UnstructuredS3TableSink` call `provision()` (create namespace/table if missing) and then append rows, which updates table metadata locations and writes table data.
- `run_index_cluster()` appends `flow_run_id` to sink table names for test/isolation runs, so IAM should allow matching wildcard table names in the `data_index` namespace.
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
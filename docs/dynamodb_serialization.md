# DynamoDB `to_dynamodb()` Implementation Requirements

Implement a `to_dynamodb()` method for our Python models that converts arbitrary Python objects into a structure that can be safely persisted using boto3 DynamoDB APIs such as `put_item()`.

The output must always be DynamoDB-compatible and require no additional preprocessing before being passed to boto3.

---

## Design Goals

- Pure function.
- No mutation of the original object.
- Deterministic output.
- Strong typing.
- Comprehensive docstrings.
- Fully recursive conversion.
- Clear and actionable validation errors.
- Output must always be safe to pass directly to DynamoDB.

Example:

```python
item = MyModel(...)

ddb_item = item.to_dynamodb()

table.put_item(Item=ddb_item)
```

---

## Method Signature

```python
to_dynamodb(
    include_nulls: bool = False,
    partition_key: str | None = None,
    sort_key: str | None = None,
)
```

---

# Number Handling

## Floats

DynamoDB does not support Python `float`.

Recursively convert all floats to:

```python
Decimal(str(value))
```

This must apply to:

- top-level values
- nested dictionaries
- lists
- tuples
- sets

Example:

```python
{"price": 19.95}
```

Becomes:

```python
{"price": Decimal("19.95")}
```

### Precision

Preserve precision exactly by using:

```python
Decimal(str(value))
```

Do not use:

```python
Decimal(value)
```

---

## NaN and Infinity

Detect and handle:

```python
float("nan")
float("inf")
float("-inf")
```

Default behaviour:

```python
None
```

before subsequent null handling occurs.

This is required because DynamoDB cannot store NaN or Infinity values.

---

# Nullable Attribute Handling

DynamoDB supports NULL attributes, but our preferred modelling approach is:

> Optional attributes should generally be omitted rather than stored as NULL.

## Default Behaviour

When:

```python
include_nulls = False
```

all values that resolve to:

```python
None
```

must be removed entirely from the output.

Example:

```python
{"name": "Tom", "middle_name": None}
```

Becomes:

```python
{"name": "Tom"}
```

---

## Preserve NULL Values

When:

```python
include_nulls = True
```

retain `None` values.

Example:

```python
{"name": "Tom", "middle_name": None}
```

Remains:

```python
{"name": "Tom", "middle_name": None}
```

allowing boto3 to serialize the field as:

```json
{
  "NULL": true
}
```

---

# Unsupported Python Types

The converter must recursively normalize unsupported Python objects.

## datetime

Convert:

```python
datetime
```

to:

```python
value.isoformat()
```

---

## date

Convert:

```python
date
```

to:

```python
value.isoformat()
```

---

## UUID

Convert:

```python
UUID
```

to:

```python
str(value)
```

---

## Enum

Convert:

```python
Enum
```

to:

```python
value.value
```

---

## pathlib.Path

Convert:

```python
Path
```

to:

```python
str(value)
```

---

# Collection Support

Support arbitrary nesting of:

- dict
- list
- tuple
- set

Example:

```python
{"facility": {"config": {"threshold": 1.5}}}
```

Must become:

```python
{"facility": {"config": {"threshold": Decimal("1.5")}}}
```

---

# Set Handling

DynamoDB sets must be homogeneous.

## Empty Sets

Empty sets are invalid in DynamoDB.

Remove them entirely.

Example:

```python
{"tags": set()}
```

Result:

```python
{}
```

or the attribute is omitted from the containing structure.

---

## Homogeneous Sets

Allowed:

```python
{"a", "b", "c"}
```

Allowed:

```python
{1, 2, 3}
```

Allowed:

```python
{Decimal("1"), Decimal("2")}
```

---

## Mixed-Type Sets

Reject mixed-type sets.

Example:

```python
{1, "a"}
```

Must raise a validation error.

---

# NumPy Support

Convert NumPy scalar types into native or DynamoDB-compatible types.

## Integers

Convert:

```python
np.int64
np.int32
np.int16
```

to:

```python
int
```

---

## Floats

Convert:

```python
np.float64
np.float32
```

to:

```python
Decimal(str(value))
```

---

## Boolean

Convert:

```python
np.bool_
```

to:

```python
bool
```

---

# Pandas Support

Treat all pandas missing values as:

```python
None
```

Examples:

```python
pd.NA
numpy.nan
NaT
```

should become:

```python
None
```

before standard null handling is applied.

---

# Key Validation

Provide optional validation of DynamoDB keys.

Example:

```python
to_dynamodb(partition_key="pk", sort_key="sk")
```

## Partition Key Rules

The partition key must:

- exist
- not be None
- not be an empty string

Invalid:

```python
{"pk": None}
```

Invalid:

```python
{"pk": ""}
```

---

## Sort Key Rules

If configured, the sort key must:

- exist
- not be None
- not be an empty string

Invalid:

```python
{"pk": "USER#1", "sk": None}
```

---

# Error Handling

Do not silently drop invalid values except:

- `None` when `include_nulls=False`
- empty sets

Everything else should produce a clear exception.

Raise:

```python
ValueError
```

with the full attribute path included.

Example:

```text
Invalid DynamoDB value at:
facility.config.threshold

Reason:
mixed-type set detected
```

Another example:

```text
Invalid DynamoDB value at:
facilities[4].reading

Reason:
NaN value is not supported
```

This path-aware validation is required to make production data quality issues easy to diagnose.

---

# Recursive Behaviour

The implementation must recurse through:

- nested dictionaries
- nested lists
- nested tuples
- nested sets
- complex mixed structures

Example:

```python
{"facility": {"readings": [{"temperature": 18.2}]}}
```

Must become:

```python
{"facility": {"readings": [{"temperature": Decimal("18.2")}]}}
```

---

# Unit Test Coverage

Include tests covering:

- float conversion
- nested float conversion
- datetime conversion
- date conversion
- UUID conversion
- Enum conversion
- `None` removal
- NULL preservation
- empty set removal
- homogeneous string sets
- homogeneous numeric sets
- mixed-type set rejection
- NumPy integer conversion
- NumPy float conversion
- NumPy boolean conversion
- pandas `NaN` handling
- pandas `NA` handling
- partition key validation
- sort key validation
- nested document conversion
- path-aware exceptions

---

# Success Criteria

The resulting object must:

- Be safe to pass directly to boto3 `put_item()`.
- Contain no Python floats.
- Contain no unsupported Python types.
- Contain no invalid DynamoDB set structures.
- Correctly handle optional attributes.
- Produce deterministic results.
- Provide actionable validation messages when conversion fails.
- Support arbitrarily nested document-style payloads commonly stored in DynamoDB.
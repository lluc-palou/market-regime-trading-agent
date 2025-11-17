# LOB Data Directory

This directory contains raw LOB (Limit Order Book) .parquet files that can be uploaded to S3 using the dedicated upload script.

## Purpose

- Store raw LOB data files locally before uploading to S3
- Separate from the processed splits (pipeline stages 15-16)
- Files uploaded to S3 under the `lob-data` prefix (not `processed-splits`)

## Usage

1. Place your .parquet files in this directory
2. Run the upload script:
   ```bash
   python scripts/upload_lob_data_to_s3.py
   ```

3. Optional: Use custom run ID or source directory:
   ```bash
   python scripts/upload_lob_data_to_s3.py --run-id my_custom_name
   python scripts/upload_lob_data_to_s3.py --source-dir /path/to/other/dir
   ```

## S3 Structure

Files are uploaded to:
```
s3://your-drl-lob-bucket/lob-data/{run_id}/
  ├── file1.parquet
  ├── file2.parquet
  └── manifest.json
```

## Notes

- `.parquet` files in this directory are gitignored (not committed to repository)
- Upload manifests are saved locally in `artifacts/lob_data_uploads/`
- Each upload run is timestamped for versioning

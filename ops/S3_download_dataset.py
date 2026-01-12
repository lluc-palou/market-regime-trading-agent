"""
S3 Dataset Download Script

Downloads data from S3 bucket based on mode:
- Train mode: Downloads 'data/' folder for training
- Test mode: Downloads 'test_data/' folder for testing

Usage:
    python ops/S3_download_dataset.py --mode train
    python ops/S3_download_dataset.py --mode test
"""

import boto3
import os
import argparse

BUCKET_NAME = "lluc-tfg-data"

def download_folder(bucket_name, prefix, local_dir):
    """
    Download all files from an S3 folder to a local directory.

    Args:
        bucket_name: Name of the S3 bucket
        prefix: S3 folder prefix (e.g., 'data/' or 'test_data/')
        local_dir: Local directory where files will be saved
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    print(f"Downloading from s3://{bucket_name}/{prefix} -> {local_dir}")

    file_count = 0
    for obj in bucket.objects.filter(Prefix=prefix):
        # Remove the folder prefix to reconstruct directory structure locally
        relative_path = obj.key[len(prefix):]

        # Skip directories (S3 lists them as objects ending with "/")
        if obj.key.endswith("/"):
            continue

        local_path = os.path.join(local_dir, relative_path)

        # Create local folders if necessary
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  Downloading {obj.key} -> {local_path}")
        bucket.download_file(obj.key, local_path)
        file_count += 1

    print(f"Download completed: {file_count} files downloaded")

def main():
    parser = argparse.ArgumentParser(
        description='Download dataset from S3 bucket',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download training data
  python ops/S3_download_dataset.py --mode train

  # Download test data
  python ops/S3_download_dataset.py --mode test
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        required=True,
        help='Mode: train (downloads data/) or test (downloads test_data/)'
    )

    args = parser.parse_args()

    # Set S3 prefix and local directory based on mode
    if args.mode == 'train':
        s3_prefix = "data/"
        local_dir = "data/"
        print("=" * 80)
        print("TRAIN MODE: Downloading training data")
        print("=" * 80)
    else:  # test mode
        s3_prefix = "test_data/"
        local_dir = "test_data/"
        print("=" * 80)
        print("TEST MODE: Downloading test data")
        print("=" * 80)

    # Download the folder
    download_folder(BUCKET_NAME, s3_prefix, local_dir)

if __name__ == "__main__":
    main()
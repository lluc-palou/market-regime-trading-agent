import boto3
import os

BUCKET_NAME = "lluc-tfg-data"
PREFIX = "data/"       # S3 folder you want to download
LOCAL_DIR = "data/"    # local directory where files will be saved

def download_folder(bucket_name, prefix, local_dir):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        # Remove the folder prefix to reconstruct directory structure locally
        relative_path = obj.key[len(prefix):]

        # Skip directories (S3 lists them as objects ending with "/")
        if obj.key.endswith("/"):
            continue
        
        local_path = os.path.join(local_dir, relative_path)

        # Create local folders if necessary
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {obj.key} -> {local_path}")
        bucket.download_file(obj.key, local_path)

    print("Download completed")

if __name__ == "__main__":
    download_folder(BUCKET_NAME, PREFIX, LOCAL_DIR)
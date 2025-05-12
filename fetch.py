import pandas as pd
from google.cloud import storage
from io import BytesIO

BUCKET_NAME = "ulta-dash-app-v2.appspot.com"

storage_client = storage.Client()

def load_csv_from_gcs(filename):
    """Load a single CSV file from GCS into a Pandas DataFrame."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        print(f"Loaded {filename} successfully.")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()  # Safe fallback if fail

def get_cleaned_makeup_products():
    return load_csv_from_gcs("cleaned_makeup_products.csv")

def get_face_df():
    return load_csv_from_gcs("face_df.csv")

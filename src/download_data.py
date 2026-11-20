
import os
import urllib.request
import zipfile

# Dataset download URL
url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

# Create raw data folder
os.makedirs("data/raw", exist_ok=True)

zip_path = "data/raw/online_retail.zip"
excel_path = "data/raw/Online Retail.xlsx"

if os.path.exists(excel_path):
    print("Dataset already exists!")
else:
    print("Downloading dataset...")

    urllib.request.urlretrieve(url, zip_path)

    print("Download completed!")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data/raw")

    os.remove(zip_path)

    print("Dataset extracted successfully!")

# Download Tools

import requests
import shutil
import os


def download_file(url, output_folder=".", output_filename=None):
    filename = output_filename if output_filename else url.split('/')[-1]
    filepath = os.path.join(output_folder, filename)
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return filepath

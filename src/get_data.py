"""
Download and extract data. At least if we want it to work this way.
Currently the data already is here locally, so this step does nothing.
"""
import urllib.request
import zipfile

# URL = '<URL>'
URL = None

EXTRACT_DIR = "dataset"

if URL is not None:
    zip_path, _ = urllib.request.urlretrieve(URL)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(EXTRACT_DIR)
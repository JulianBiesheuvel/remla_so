"""
Download and extract data. At least if we want it to work this way.
Currently the data already is here locally, so this step does nothing.
"""
import urllib.request
import zipfile

from . import conf

# URL = '<URL>'
URL = None

if URL is not None:
    zip_path, _ = urllib.request.urlretrieve(URL)  # nosec
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(conf.RAW_DATA_DIR)

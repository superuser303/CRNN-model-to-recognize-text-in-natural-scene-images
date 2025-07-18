import requests
import zipfile
import io
import os

url = "https://cvit.iiit.ac.in/files/IIIT5K.zip"
out_path = "data/IIIT5K"

os.makedirs(out_path, exist_ok=True)
print("Downloading IIIT5K...")
r = requests.get(url)

if r.status_code != 200:
    print("Failed to download IIIT5K.zip. Status code:", r.status_code)
    print("Response:", r.text[:500])  # Print first 500 chars of response for debug
    exit(1)

try:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(out_path)
    print("Done! Extracted to", out_path)
except zipfile.BadZipFile:
    print("Downloaded file is not a valid zip. Check the URL or your network.")
    with open("downloaded_file.html", "wb") as f:
        f.write(r.content)
    print("Saved the response to downloaded_file.html for inspection.")
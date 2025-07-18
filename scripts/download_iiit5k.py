import requests
import zipfile
import io
import os

url = "https://cvit.iiit.ac.in/files/IIIT5K.zip"
out_path = "data/IIIT5K"

os.makedirs(out_path, exist_ok=True)
print("Downloading IIIT5K...")
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(out_path)
print("Done! Extracted to", out_path)
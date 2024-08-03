import zipfile
with zipfile.ZipFile("road_data.zip", 'r') as zip_ref:
    zip_ref.extractall()
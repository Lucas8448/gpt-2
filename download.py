import os
import requests

model_dir = "gpt2"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

files = {
    "config.json": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json",
    "pytorch_model.bin": "https://cdn.huggingface.co/gpt2-pytorch_model.bin",
    "vocab.json": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
    "merges.txt": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
}

for file_name, file_url in files.items():
    local_file_path = os.path.join(model_dir, file_name)
    if not os.path.exists(local_file_path):
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            with open(local_file_path, "wb") as local_file:
                for chunk in response.iter_content(chunk_size=8192):
                    local_file.write(chunk)
        print(f"Downloaded {file_name}")
    else:
        print(f"{file_name} already exists")

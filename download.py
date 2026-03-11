import kagglehub

# Download latest version
path = kagglehub.dataset_download("ubamba98/deberta-v3-large-v18-margin-focused-v2")

print("Path to dataset files:", path)
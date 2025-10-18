import kagglehub

# Download latest version
housing_data = kagglehub.dataset_download("camnugent/california-housing-prices")
cancer_data = kagglehub.dataset_download("yasserh/breast-cancer-dataset")


print("Path to dataset files:", housing_data, "\n", cancer_data)
def download_insect_data():
    """Download InsectSound1000 dataset from Kaggle using kagglehub"""
    import kagglehub
    from pathlib import Path
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("hesi0ne/insectsound1000")
    print("Path to dataset files:", path)

if __name__ == "__main__":
    download_insect_data()

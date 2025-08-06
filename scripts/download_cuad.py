"""
Download and setup CUAD dataset from GitHub
"""
import os
import urllib.request
import zipfile
import json
from pathlib import Path
import shutil

def download_cuad():
    """Download CUAD dataset from GitHub"""
    print("Downloading CUAD dataset...")
    
    # Create data directory
    data_dir = Path("data/raw/cuad")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different possible URLs
    urls_to_try = [
        "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
        "https://github.com/TheAtticusProject/cuad/raw/master/data.zip"
    ]
    
    for url in urls_to_try:
        try:
            print(f"Trying URL: {url}")
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            with urllib.request.urlopen(req) as response:
                with open("cuad_data.zip", 'wb') as f:
                    f.write(response.read())
            
            print("Download successful. Extracting...")
            
            with zipfile.ZipFile("cuad_data.zip", 'r') as zip_ref:
                zip_ref.extractall(str(data_dir))
            
            os.remove("cuad_data.zip")
            print("Extraction complete")
            return True
            
        except Exception as e:
            print(f"Failed with {url}: {e}")
            continue
    
    print("All download attempts failed.")
    print("Manual download required:")
    print("1. Visit: https://github.com/TheAtticusProject/cuad")
    print("2. Download data.zip")
    print("3. Extract to: data/raw/cuad/")
    return False

def explore_cuad():
    """Explore downloaded CUAD data"""
    print("Exploring CUAD dataset...")
    
    cuad_dir = Path("data/raw/cuad")
    
    if not cuad_dir.exists():
        print("CUAD directory not found")
        return
    
    # List files
    all_files = list(cuad_dir.rglob("*"))
    
    if not all_files:
        print("No files found in CUAD directory")
        return
    
    # Categorize files
    json_files = list(cuad_dir.rglob("*.json"))
    csv_files = list(cuad_dir.rglob("*.csv"))
    txt_files = list(cuad_dir.rglob("*.txt"))
    
    print(f"Found files:")
    print(f"  JSON: {len(json_files)}")
    print(f"  CSV: {len(csv_files)}")
    print(f"  TXT: {len(txt_files)}")
    
    # Show file details
    for file_path in all_files[:10]:  # First 10 files
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_path.relative_to(cuad_dir)} ({size_mb:.1f} MB)")
    
    # Quick data check
    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                print(f"First JSON contains {len(data['data'])} records")
        except Exception as e:
            print(f"Error reading JSON: {e}")

if __name__ == "__main__":
    print("CUAD Dataset Setup")
    print("=" * 30)
    
    success = download_cuad()
    if success:
        explore_cuad()
        print("Setup complete")
    else:
        print("Setup failed - manual download required")
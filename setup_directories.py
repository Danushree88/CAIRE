# setup_directories.py
import os
from pathlib import Path

def setup_project_structure():
    """Create necessary directories for the project"""
    directories = [
        "data",
        "saved_models", 
        "test_data",
        "src/models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎯 Project structure ready!")
    print("Next steps:")
    print("1. Place your final_manual_model.pkl in saved_models/")
    print("2. Ensure cart_abandonment_featured.csv is in data/")
    print("3. Run: streamlit run admin_dashboard.py")

if __name__ == "__main__":
    setup_project_structure()
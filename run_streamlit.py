#!/usr/bin/env python
"""
Helper script to run the Streamlit app from the project root directory.
This ensures that all file paths are correctly resolved.
"""

import os
import subprocess
import sys

def main():
    # Get the absolute path to the app directory
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py")
    
    print(f"Starting Streamlit app: {app_path}")
    
    # Run streamlit with the app path
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError:
        print("Error running Streamlit app")
        return 1
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
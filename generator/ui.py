#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
import threading

def run_backend():
    """Starts the Python FastAPI Server on port 8000"""
    print("⚡ Starting Python Backend (Port 8000)...")
    # We use sys.executable to ensure we use the same python env
    subprocess.run([sys.executable, "server.py"], cwd=os.getcwd())

def run_frontend(ui_dir):
    """Starts the Next.js Frontend on port 3000"""
    print(f"⚛️  Starting React Frontend (Port 3000)...")
    
    is_windows = sys.platform == "win32"
    npm_cmd = "npm.cmd" if is_windows else "npm"
    
    if not shutil.which(npm_cmd) and not is_windows:
         npm_cmd = "npm"

    subprocess.run([npm_cmd, "run", "dev"], cwd=ui_dir)

def main():
    script_dir = Path(__file__).resolve().parent
    ui_dir = script_dir / "ui"

    # Dependency Checks
    if not (ui_dir / "node_modules").exists():
        print("📦 Installing UI dependencies...")
        subprocess.run(["npm", "install"], cwd=ui_dir, shell=(sys.platform=="win32"))

    try:
        # Run Backend in a separate thread so it doesn't block
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()

        # Give backend a second to warm up
        time.sleep(2)

        # Run Frontend in main thread
        run_frontend(ui_dir)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()
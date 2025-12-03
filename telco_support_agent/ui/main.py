"""Telco Support Agent UI - Single file entry point for Databricks Apps."""

import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
backend_dir = current_dir / "backend"
sys.path.insert(0, str(backend_dir))


def main():
    """Main entry point."""
    try:
        import uvicorn
        from app.main import app

        port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))

        print("Starting Telco Support Agent UI")
        print(f"Working directory: {os.getcwd()}")
        print(f"Port: {port}")
        print(f"Backend path: {backend_dir}")

        static_dir = current_dir / "static"
        if static_dir.exists():
            file_count = len(list(static_dir.glob("*")))
            print(f"Static files: {file_count} files found")
        else:
            print("WARNING: No static directory found")

        uvicorn.run(
            app,
            host="0.0.0.0",  # noqa: S104
            port=port,
            log_level="info",
        )

    except Exception as e:
        print(f"Error starting app: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Development server script for Telco Support Agent UI backend."""

import sys
from pathlib import Path

# add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    import uvicorn
    from app.config import get_settings

    settings = get_settings()

    print("=" * 50)
    print("Starting Telco Support Agent UI Backend")
    print("=" * 50)
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Databricks Endpoint: {settings.databricks_endpoint}")
    print("=" * 50)
    print()
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Chat API: http://localhost:8000/api/chat")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    # check if .env file exists
    env_file = current_dir / ".env"
    if not env_file.exists():
        print("WARNING: .env file not found!")
        print("   Copy .env.example to .env and configure your settings")
        print("=" * 50)

    # check if Databricks token is set
    if (
        not settings.databricks_token
        or settings.databricks_token == "databricks_token_here"  # noqa: S105
    ):
        print("ERROR: DATABRICKS_TOKEN not configured!")
        print("   Please set your Databricks token in the .env file")
        sys.exit(1)

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.environment == "development",
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""Run the AI News Agent MCP server."""

import os
import sys

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(project_dir, ".env"))

# Run the server
from orchestration.server import main

if __name__ == "__main__":
    main()

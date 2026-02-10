#!/usr/bin/env python3
"""
Environment verification script for AI News Agent.
Checks that all dependencies are installed and environment is configured.
"""
import sys
import os

def check_imports():
    """Check that all required packages are installed."""
    required_packages = {
        'openai': 'openai',
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'dotenv': 'python-dotenv',
        'feedparser': 'feedparser',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (missing)")
            missing.append(package_name)
    
    return len(missing) == 0, missing

def check_files():
    """Check that required files exist."""
    required_files = {
        'sources.txt': 'RSS feed sources file',
        'agent.py': 'Main agent script',
        'summarizer.py': 'Summarizer module',
    }
    
    missing = []
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✓ {file} ({description})")
        else:
            print(f"✗ {file} ({description}) - missing")
            missing.append(file)
    
    return len(missing) == 0, missing

def check_env():
    """Check environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    api_ok = api_key and api_key != "your-openai-api-key"
    email_ok = smtp_username and smtp_password
    
    if api_ok:
        print("✓ OPENAI_API_KEY is set")
    else:
        print("⚠ OPENAI_API_KEY is not set (summaries will use fallback)")
    
    if email_ok:
        print(f"✓ Email configured (sending to: {os.getenv('EMAIL_TO', 'robert.burden@gmail.com')})")
    else:
        print("⚠ Email not configured (set SMTP_USERNAME and SMTP_PASSWORD in .env)")
    
    return api_ok and email_ok

def main():
    print("AI News Agent - Environment Check\n")
    print("Checking dependencies...")
    deps_ok, missing_deps = check_imports()
    print()
    
    print("Checking files...")
    files_ok, missing_files = check_files()
    print()
    
    print("Checking environment...")
    env_ok = check_env()
    print()
    
    if not deps_ok:
        print("❌ Missing dependencies. Install with:")
        print("   pip install -r requirements.txt")
        return 1
    
    if not files_ok:
        print("❌ Missing required files.")
        return 1
    
    if env_ok:
        print("✅ Environment is fully configured!")
    else:
        print("⚠️  Environment is mostly configured, but API key is missing.")
        print("   The agent will work but summaries will use fallback text.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

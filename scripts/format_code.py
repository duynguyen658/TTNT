import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print("=" * 60)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("CODE FORMATTING TOOL")
    print("=" * 60)

    # Check if tools are installed
    tools = {"black": "black --version", "isort": "isort --version", "flake8": "flake8 --version"}

    print("\n📋 Kiểm tra tools...")
    for tool, cmd in tools.items():
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"  ✅ {tool} đã cài đặt")
        except:
            print(f"  ❌ {tool} chưa cài đặt")
            print(f"     Cài đặt: pip install {tool}")
            return

    # Format code
    success = True

    # 1. Sort imports
    success &= run_command("isort . --profile black --line-length 100", "Sắp xếp imports với isort")

    # 2. Format with Black
    success &= run_command("black . --line-length 100", "Format code với Black")

    # 3. Check with Flake8
    success &= run_command(
        "flake8 . --max-line-length=100 --extend-ignore=E203,W503,E226,E722,F401,F541,C901 --max-complexity=30",
        "Kiểm tra code với Flake8",
    )

    print("\n" + "=" * 60)
    if success:
        print("✅ Hoàn thành! Code đã được format và kiểm tra")
    else:
        print("⚠️  Có một số lỗi, vui lòng kiểm tra lại")
    print("=" * 60)


if __name__ == "__main__":
    main()

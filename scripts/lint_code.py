import subprocess
import sys
from pathlib import Path


def run_linter(cmd, description):
    """Chạy linter và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print("=" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("CODE LINTING TOOL")
    print("=" * 60)

    # Check if flake8 is installed
    try:
        subprocess.run("flake8 --version", shell=True, check=True, capture_output=True)
    except:
        print("❌ Flake8 chưa cài đặt")
        print("👉 Cài đặt: pip install flake8")
        return

    # Run linters
    all_passed = True

    # Flake8
    all_passed &= run_linter(
        "flake8 . --max-line-length=100 --extend-ignore=E203,W503,E226,E722,F401,F541,C901 --max-complexity=30 --statistics",
        "Kiểm tra code với Flake8",
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Tất cả checks đã pass!")
    else:
        print("⚠️  Có một số vấn đề, vui lòng sửa lại")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

# Code Quality Tools Guide

Hướng dẫn sử dụng các công cụ chất lượng code đã được cấu hình sẵn.

## 📦 Cài Đặt

### Python Tools

```bash
# Cài đặt development dependencies
pip install -r requirements-dev.txt

# Hoặc cài từng tool
pip install black flake8 isort pre-commit
```

### JavaScript Tools (nếu có JS code)

```bash
# Cài đặt npm packages
npm install

# Hoặc cài từng tool
npm install --save-dev eslint prettier eslint-config-prettier eslint-plugin-prettier
```

## 🐍 Python Tools

### Black - Code Formatter

**Tự động format code Python theo chuẩn PEP 8**

```bash
# Format tất cả files
black .

# Format một file cụ thể
black path/to/file.py

# Check (không format, chỉ báo lỗi)
black --check .

# Format với line length tùy chỉnh
black --line-length 100 .
```

**Config**: Đã cấu hình trong `pyproject.toml`

### isort - Import Sorter

**Tự động sắp xếp imports theo chuẩn**

```bash
# Sort imports
isort .

# Sort một file
isort path/to/file.py

# Check only
isort --check-only .

# Với profile Black (khuyến nghị)
isort . --profile black
```

**Config**: Đã cấu hình trong `pyproject.toml` với profile Black

### Flake8 - Linter

**Kiểm tra code style và lỗi tiềm ẩn**

```bash
# Lint tất cả files
flake8 .

# Lint một file
flake8 path/to/file.py

# Với statistics
flake8 . --statistics

# Ignore specific errors (đã cấu hình sẵn)
flake8 . --extend-ignore=E203,W503,E226,E722,F401,F541
```

**Config**: Đã cấu hình trong `.flake8`

**Lưu ý**: Các lỗi docstring (D*) đã được ignore vì không bắt buộc cho dự án này. Nếu muốn kiểm tra docstring, có thể cài `flake8-docstrings` và bỏ ignore các lỗi D*.

### Mypy - Type Checker (Optional)

```bash
# Type check
mypy .

# Check một file
mypy path/to/file.py
```

## 📝 JavaScript Tools

### ESLint - JavaScript Linter

```bash
# Lint tất cả JS files
npm run lint:js

# Lint và tự động fix
npm run lint:js:fix

# Hoặc dùng trực tiếp
npx eslint . --ext .js,.jsx,.ts,.tsx
```

### Prettier - Code Formatter

```bash
# Format tất cả files
npm run format:js

# Check only
npm run format:js:check

# Hoặc dùng trực tiếp
npx prettier --write "**/*.{js,jsx,ts,tsx,json,css,html,md,yaml}"
```

## 🚀 Quick Commands

### Format tất cả code (Python)

```bash
# Sử dụng script tự động
python scripts/format_code.py

# Hoặc thủ công
isort . --profile black
black .
flake8 .
```

### Lint code (Python)

```bash
# Sử dụng script
python scripts/lint_code.py

# Hoặc thủ công
flake8 . --statistics
```

### Format tất cả code (JavaScript)

```bash
npm run format:js
npm run lint:js:fix
```

## 🔧 Pre-commit Hooks

Tự động chạy các tools trước khi commit:

### Setup

```bash
# Cài đặt pre-commit
pip install pre-commit

# Cài đặt hooks
pre-commit install

# Chạy thử trên tất cả files
pre-commit run --all-files
```

### Sử dụng

Sau khi setup, mỗi lần `git commit`, các hooks sẽ tự động:
- Format code với Black
- Sort imports với isort
- Lint với Flake8
- Format JS với Prettier
- Lint JS với ESLint

### Bỏ qua hooks (không khuyến nghị)

```bash
git commit --no-verify
```

## 📋 Workflow Khuyến Nghị

### Trước khi commit:

1. **Format code**:
   ```bash
   python scripts/format_code.py
   ```

2. **Lint code**:
   ```bash
   python scripts/lint_code.py
   ```

3. **Hoặc dùng pre-commit** (tự động):
   ```bash
   git add .
   git commit -m "Your message"
   # Pre-commit sẽ tự động chạy
   ```

### Trong CI/CD:

Thêm vào pipeline:

```yaml
# Example GitHub Actions
- name: Format check
  run: |
    black --check .
    isort --check-only .
    flake8 .

- name: Lint check
  run: |
    flake8 . --statistics
```

## ⚙️ Configuration Files

- **Black**: `pyproject.toml` (section `[tool.black]`)
- **isort**: `pyproject.toml` (section `[tool.isort]`)
- **Flake8**: `.flake8`
- **Pre-commit**: `.pre-commit-config.yaml`
- **ESLint**: `.eslintrc.json`
- **Prettier**: `.prettierrc.json`

## 🎯 Best Practices

1. **Luôn format code trước khi commit**
   ```bash
   black . && isort .
   ```

2. **Fix linting errors trước khi push**
   ```bash
   flake8 .  # Xem lỗi
   # Sửa lỗi
   flake8 .  # Kiểm tra lại
   ```

3. **Sử dụng pre-commit hooks** để tự động hóa

4. **Trong team**: Đảm bảo mọi người dùng cùng config

5. **Trong CI/CD**: Chạy checks tự động

## 🔍 Ignore Files

Các file/thư mục đã được ignore:
- `runs/` - Training results
- `models/*.pt` - Model files
- `data/` - Dataset files
- `__pycache__/`, `.venv/`, `venv/` - Python cache
- `node_modules/` - Node modules

## 📚 Tài Liệu Tham Khảo

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [ESLint Documentation](https://eslint.org/)
- [Prettier Documentation](https://prettier.io/)

## ❓ Troubleshooting

### Lỗi: "command not found"
```bash
# Đảm bảo đã cài đặt
pip install black flake8 isort
```

### Lỗi: "No module named 'black'"
```bash
# Cài trong virtual environment
source venv/bin/activate  # hoặc .venv/Scripts/activate trên Windows
pip install -r requirements-dev.txt
```

### Lỗi: "Pre-commit hook failed"
```bash
# Chạy thủ công để xem lỗi
pre-commit run --all-files

# Hoặc skip hook (tạm thời)
git commit --no-verify
```

### ESLint không chạy
```bash
# Đảm bảo đã cài
npm install

# Hoặc cài global
npm install -g eslint prettier
```

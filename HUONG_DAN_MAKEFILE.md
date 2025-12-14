# 📖 Hướng Dẫn Sử Dụng Makefile

Makefile giúp bạn chạy các lệnh phổ biến một cách nhanh chóng và dễ nhớ.

## 🚀 Cài Đặt

### Windows

Makefile cần **Make** tool. Có 2 cách:

**Cách 1: Dùng Chocolatey (Khuyến nghị)**

```powershell
choco install make
```

**Cách 2: Dùng Git Bash**

- Git Bash đã có sẵn `make`
- Mở Git Bash và chạy lệnh

**Cách 3: Dùng WSL (Windows Subsystem for Linux)**

```bash
# Trong WSL
sudo apt-get install make
```

### Linux/Mac

Đã có sẵn `make`, không cần cài đặt.

## 📋 Các Lệnh Cơ Bản

### Xem Tất Cả Lệnh Có Sẵn

```bash
make help
```

Hiển thị danh sách tất cả lệnh và mô tả.

### Cài Đặt Dependencies

**Cài đặt production dependencies:**

```bash
make install
```

Tương đương: `pip install -r requirements.txt`

**Cài đặt development dependencies:**

```bash
make install-dev
```

Tương đương:

- `pip install -r requirements-dev.txt`
- `npm install`

## 🎨 Format Code

### Format Code (Tự Động Sửa)

```bash
make format
```

- Format Python code (isort + black)
- Format JavaScript code (prettier)

### Kiểm Tra Format (Không Sửa)

```bash
make format-check
```

Chỉ kiểm tra xem code đã đúng format chưa, không tự động sửa.

## 🔍 Lint Code

### Lint Code (Kiểm Tra Lỗi)

```bash
make lint
```

- Lint Python code (flake8)
- Lint JavaScript code (eslint)

### Lint và Tự Động Fix JavaScript

```bash
make lint-fix
```

Tự động sửa các lỗi JavaScript có thể sửa được.

## ✅ Kiểm Tra Tất Cả

### Chạy Tất Cả Checks

```bash
make check
```

Chạy cả `format-check` và `lint` - kiểm tra format và lint.

## 🧪 Test

### Chạy Tests

```bash
make test
```

Tương đương: `pytest`

## 🧹 Dọn Dẹp

### Xóa Cache và Build Files

```bash
make clean
```

Xóa:

- `__pycache__/` folders
- `.pytest_cache/`
- `.mypy_cache/`
- `*.pyc`, `*.pyo` files
- `build/`, `dist/`, `*.egg-info/`

## ⚙️ Setup Pre-commit

### Cài Đặt Pre-commit Hooks

```bash
make setup-precommit
```

- Cài đặt pre-commit hooks
- Chạy hooks trên tất cả files

## 🎯 Làm Tất Cả

### Cài Đặt, Format và Lint Tất Cả

```bash
make all
```

Chạy: `install-dev` → `format` → `lint`

## 📝 Ví Dụ Sử Dụng

### Workflow Thông Thường

**1. Lần đầu setup project:**

```bash
make install-dev        # Cài đặt dependencies
make setup-precommit    # Setup pre-commit hooks
```

**2. Trước khi commit code:**

```bash
make format            # Format code
make lint              # Kiểm tra lỗi
# hoặc
make check             # Kiểm tra format + lint
```

**3. Khi code bị lỗi format:**

```bash
make format            # Tự động sửa format
```

**4. Dọn dẹp trước khi build:**

```bash
make clean             # Xóa cache
```

## 🔧 Tùy Chỉnh

Nếu muốn thay đổi cấu hình, sửa file `Makefile`:

- **Line length**: Mặc định là 100, có thể thay đổi trong các lệnh `format` và `lint`
- **Flake8 ignore**: Có thể thêm/bớt ignore rules trong lệnh `lint`
- **Thêm lệnh mới**: Thêm target mới vào Makefile

## ⚠️ Lưu Ý

1. **Windows**: Cần cài `make` hoặc dùng Git Bash/WSL
2. **npm commands**: Một số lệnh cần `package.json` và `node_modules`
3. **Python venv**: Nên chạy trong virtual environment
4. **Pre-commit**: Cần cài đặt `pre-commit` package trước

## 🆘 Troubleshooting

### Lỗi: `make: command not found`

- **Windows**: Cài `make` qua Chocolatey hoặc dùng Git Bash
- **Linux/Mac**: Đã có sẵn, nếu không có thì cài: `sudo apt-get install make` (Linux)

### Lỗi: `npm: command not found`

- Cài Node.js: https://nodejs.org/
- Hoặc bỏ qua các lệnh liên quan đến JavaScript

### Lỗi: `pip: command not found`

- Cài Python và pip
- Hoặc dùng `python -m pip` thay vì `pip`

## 📚 Tài Liệu Tham Khảo

- **Makefile syntax**: https://www.gnu.org/software/make/manual/
- **Black formatter**: https://black.readthedocs.io/
- **Flake8 linter**: https://flake8.pycqa.org/
- **Prettier**: https://prettier.io/

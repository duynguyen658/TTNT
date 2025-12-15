import os
import re
from pathlib import Path

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_header(text: str):
    """Print header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")


def print_success(text: str):
    """Print success"""
    print(f"{GREEN}✅ {text}{RESET}")


def print_error(text: str):
    """Print error"""
    print(f"{RED}❌ {text}{RESET}")


def print_warning(text: str):
    """Print warning"""
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_info(text: str):
    """Print info"""
    print(f"{CYAN}ℹ️  {text}{RESET}")


def check_providers_file():
    """Kiểm tra file providers.ts"""
    print_header("Kiểm Tra Provider Configuration")

    providers_path = Path("FE/ai-chatbot-main/lib/ai/providers.ts")

    if not providers_path.exists():
        print_error(f"Không tìm thấy file: {providers_path}")
        return False

    content = providers_path.read_text(encoding="utf-8")

    # Check for Vercel AI Gateway
    has_gateway = "gateway.languageModel" in content or "@ai-sdk/gateway" in content

    # Check for Python backend
    has_python_backend = "PYTHON_API_URL" in content or "plantDiagnosis" in content

    # Check models
    models = []
    if '"chat-model"' in content:
        models.append("chat-model (Grok Vision)")
    if '"chat-model-reasoning"' in content:
        models.append("chat-model-reasoning (Grok Reasoning)")
    if '"plant-disease-model"' in content:
        models.append("plant-disease-model (Plant Disease AI)")

    print_info(f"File: {providers_path}")

    if has_gateway:
        print_warning("⚠️  PHÁT HIỆN: Đang sử dụng Vercel AI Gateway")
        print_warning("   → Các model mặc định sẽ gọi model của người tạo gốc")

    if has_python_backend:
        print_success("✅ PHÁT HIỆN: Có kết nối đến Python backend")
        print_success("   → Model 'Plant Disease AI' sẽ dùng backend của bạn")

    print_info(f"\nCác models được định nghĩa:")
    for model in models:
        if "plant-disease" in model:
            print_success(f"  - {model} → Backend của bạn ✅")
        else:
            print_warning(f"  - {model} → Vercel AI Gateway ⚠️")

    return True


def check_plant_diagnosis_tool():
    """Kiểm tra tool plantDiagnosis"""
    print_header("Kiểm Tra Plant Diagnosis Tool")

    tool_path = Path("FE/ai-chatbot-main/lib/ai/tools/plant-diagnosis.ts")

    if not tool_path.exists():
        print_error(f"Không tìm thấy file: {tool_path}")
        return False

    content = tool_path.read_text(encoding="utf-8")

    # Check PYTHON_API_URL
    has_python_url = "PYTHON_API_URL" in content
    python_url_match = re.search(r'PYTHON_API_URL.*?=.*?["\']([^"\']+)["\']', content)

    if has_python_url:
        print_success("✅ Tool plantDiagnosis có cấu hình PYTHON_API_URL")
        if python_url_match:
            url = python_url_match.group(1)
            print_info(f"   URL: {url}")
            if "localhost:8000" in url:
                print_success("   → Đang trỏ đến backend local của bạn ✅")
            else:
                print_warning(f"   → Đang trỏ đến: {url}")
    else:
        print_error("❌ Không tìm thấy PYTHON_API_URL trong tool")

    # Check fetch call
    has_fetch = "fetch" in content and "/api/chat" in content
    if has_fetch:
        print_success("✅ Tool có gọi fetch đến backend")

    return True


def check_models_file():
    """Kiểm tra file models.ts"""
    print_header("Kiểm Tra Models Configuration")

    models_path = Path("FE/ai-chatbot-main/lib/ai/models.ts")

    if not models_path.exists():
        print_error(f"Không tìm thấy file: {models_path}")
        return False

    content = models_path.read_text(encoding="utf-8")

    # Count models
    chat_model_count = content.count('id: "')

    print_info(f"Tổng số models: {chat_model_count}")

    # Check for plant-disease-model
    has_plant_model = '"plant-disease-model"' in content
    if has_plant_model:
        print_success("✅ Model 'Plant Disease AI' có trong danh sách")
    else:
        print_error("❌ Model 'Plant Disease AI' không có trong danh sách")

    # Check for other models
    has_chat_model = '"chat-model"' in content
    has_reasoning_model = '"chat-model-reasoning"' in content

    if has_chat_model:
        print_warning("⚠️  Model 'Grok Vision' có trong danh sách → Dùng Vercel AI Gateway")

    if has_reasoning_model:
        print_warning("⚠️  Model 'Grok Reasoning' có trong danh sách → Dùng Vercel AI Gateway")

    return True


def check_env_file():
    """Kiểm tra file .env.local"""
    print_header("Kiểm Tra Environment Variables")

    env_path = Path("FE/ai-chatbot-main/.env.local")

    if not env_path.exists():
        print_warning("⚠️  File .env.local không tồn tại")
        print_info("   Tạo file với: PYTHON_API_URL=http://localhost:8000")
        return False

    content = env_path.read_text(encoding="utf-8")

    # Check PYTHON_API_URL
    python_url_match = re.search(r"PYTHON_API_URL\s*=\s*([^\s]+)", content)
    if python_url_match:
        url = python_url_match.group(1)
        print_success(f"✅ PYTHON_API_URL đã được set: {url}")
        if "localhost:8000" in url:
            print_success("   → Trỏ đến backend local của bạn ✅")
        else:
            print_warning(f"   → Trỏ đến: {url}")
    else:
        print_error("❌ PYTHON_API_URL chưa được set trong .env.local")

    # Check AI_GATEWAY_API_KEY
    has_gateway_key = "AI_GATEWAY_API_KEY" in content
    if has_gateway_key:
        print_warning("⚠️  AI_GATEWAY_API_KEY đã được set")
        print_warning("   → Có thể đang dùng Vercel AI Gateway")
    else:
        print_success("✅ AI_GATEWAY_API_KEY chưa được set")
        print_success("   → Không dùng Vercel AI Gateway (tốt)")

    return True


def check_api_route():
    """Kiểm tra API route plant-ai"""
    print_header("Kiểm Tra API Route")

    route_path = Path("FE/ai-chatbot-main/app/api/plant-ai/route.ts")

    if not route_path.exists():
        print_error(f"Không tìm thấy file: {route_path}")
        return False

    content = route_path.read_text(encoding="utf-8")

    # Check PYTHON_API_URL
    has_python_url = "PYTHON_API_URL" in content
    if has_python_url:
        print_success("✅ Route có sử dụng PYTHON_API_URL")

    # Check fetch to backend
    has_fetch = "fetch" in content and "/api/chat" in content
    if has_fetch:
        print_success("✅ Route có gọi fetch đến backend Python")

    return True


def main():
    """Main function"""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}{'KIỂM TRA MODEL ĐANG SỬ DỤNG':^70}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    results = {}

    results["providers"] = check_providers_file()
    results["tool"] = check_plant_diagnosis_tool()
    results["models"] = check_models_file()
    results["env"] = check_env_file()
    results["api_route"] = check_api_route()

    # Summary
    print_header("TÓM TẮT")

    print_info("Kết luận:")
    print_warning("⚠️  Các model 'Grok Vision' và 'Grok Reasoning' → Dùng Vercel AI Gateway")
    print_success("✅ Model 'Plant Disease AI' → Dùng backend Python của bạn")

    print_info("\nKhuyến nghị:")
    print("1. Chỉ hiển thị model 'Plant Disease AI' cho user")
    print("2. Hoặc disable các model khác nếu không cần")
    print("3. Đảm bảo PYTHON_API_URL trỏ đúng backend của bạn")

    print(f"\n{CYAN}{'='*70}{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Đã hủy{RESET}")
    except Exception as e:
        print_error(f"Lỗi: {str(e)}")
        import traceback

        traceback.print_exc()

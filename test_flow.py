"""
Script test toàn bộ flow của hệ thống Multi-Agent
"""

import asyncio
import json
import sys
from typing import Any, Dict

import requests
from colorama import Fore, Style, init

# Initialize colorama for Windows
init(autoreset=True)

# Colors
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
CYAN = Fore.CYAN
RESET = Style.RESET_ALL


def print_header(text: str):
    """Print header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✅ {text}{RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{RED}❌ {text}{RESET}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_info(text: str):
    """Print info message"""
    print(f"{CYAN}ℹ️  {text}{RESET}")


def test_backend_health(base_url: str = "http://localhost:8000") -> bool:
    """Test backend health endpoint"""
    print_header("Test 1: Backend Health Check")

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Backend đang chạy tại {base_url}")
            print_info(f"Status: {data.get('status', 'unknown')}")

            # Check agents status
            agents = data.get("agents", {})
            if agents:
                print_info(f"Số lượng agents: {len(agents)}")
                for agent_id, agent_status in agents.items():
                    status = agent_status.get("status", "unknown")
                    model = agent_status.get("model", "unknown")
                    if status == "ready":
                        print_success(f"  - {agent_id}: {model}")
                    else:
                        print_warning(f"  - {agent_id}: {status}")

            return True
        else:
            print_error(f"Backend trả về status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Không thể kết nối đến {base_url}")
        print_info("Hãy đảm bảo backend đang chạy: python api_server.py")
        return False
    except Exception as e:
        print_error(f"Lỗi: {str(e)}")
        return False


def test_api_root(base_url: str = "http://localhost:8000") -> bool:
    """Test API root endpoint"""
    print_header("Test 2: API Root Endpoint")

    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("API root endpoint hoạt động")
            print_info(f"Message: {data.get('message', 'unknown')}")
            print_info(f"Version: {data.get('version', 'unknown')}")
            return True
        else:
            print_error(f"API root trả về status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Lỗi: {str(e)}")
        return False


def test_chat_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test chat endpoint với request đơn giản"""
    print_header("Test 3: Chat Endpoint (Full Flow)")

    test_data = {
        "user_query": "Cây cà chua của tôi bị vàng lá, xin tư vấn",
        "user_context": {"plant_type": "cà chua", "location": "miền Bắc", "season": "mùa mưa"},
    }

    try:
        print_info("Gửi request đến /api/chat...")
        print_info(f"Query: {test_data['user_query']}")

        response = requests.post(
            f"{base_url}/api/chat",
            json=test_data,
            timeout=60,  # Timeout 60s vì agents cần thời gian xử lý
        )

        if response.status_code == 200:
            data = response.json()
            print_success("Chat endpoint hoạt động!")

            # Check response structure
            if "status" in data:
                print_info(f"Status: {data['status']}")

            if "final_advice" in data:
                # Check final_advice exists (variable not used but check is meaningful)
                _ = data["final_advice"]  # noqa: F841
                print_success("Có final_advice trong response")

                # Check agent results
                agent_results = data.get("agent_results", {})
                if agent_results:
                    print_info(f"Số lượng agents đã chạy: {len(agent_results)}")
                    for agent_id, result in agent_results.items():
                        if result:
                            print_success(f"  - {agent_id}: Completed")
                        else:
                            print_warning(f"  - {agent_id}: No result")

            # Print execution log
            execution_log = data.get("execution_log", [])
            if execution_log:
                print_info(f"Execution log có {len(execution_log)} entries")
                print_info("Flow execution:")
                for i, log_entry in enumerate(execution_log[:10], 1):  # Show first 10
                    print(f"  {i}. {log_entry}")

            return True
        else:
            print_error(f"Chat endpoint trả về status code: {response.status_code}")
            try:
                error_text = response.text
                print_error(f"Error: {error_text[:200]}")
            except:
                pass
            return False
    except requests.exceptions.Timeout:
        print_error("Request timeout (quá 60s)")
        print_warning("Có thể do agents xử lý lâu hoặc LLM không hoạt động")
        return False
    except Exception as e:
        print_error(f"Lỗi: {str(e)}")
        return False


def test_agents_individually(base_url: str = "http://localhost:8000") -> bool:
    """Test từng agent thông qua chat endpoint"""
    print_header("Test 4: Test Từng Agent")

    agents_to_test = [
        {
            "name": "Agent 1 - Thu thập thông tin",
            "query": "Cây cà chua bị vàng lá",
            "expected": "agent1_output",
        },
        {
            "name": "Agent 2 - Chẩn đoán hình ảnh (nếu có ảnh)",
            "query": "Phân tích hình ảnh cây bị bệnh",
            "expected": "agent2_output",
            "note": "Cần có image_data để test đầy đủ",
        },
    ]

    all_passed = True

    for agent_test in agents_to_test:
        print_info(f"Testing: {agent_test['name']}")

        test_data = {"user_query": agent_test["query"], "user_context": {}}

        try:
            response = requests.post(f"{base_url}/api/chat", json=test_data, timeout=30)

            if response.status_code == 200:
                data = response.json()
                agent_results = data.get("agent_results", {})
                expected = agent_test["expected"]

                if expected in agent_results and agent_results[expected]:
                    print_success(f"  ✅ {agent_test['name']} hoạt động")
                else:
                    print_warning(f"  ⚠️  {agent_test['name']} không có output")
                    if "note" in agent_test:
                        print_info(f"     Note: {agent_test['note']}")
            else:
                print_error(f"  ❌ {agent_test['name']} failed: {response.status_code}")
                all_passed = False

        except Exception as e:
            print_error(f"  ❌ {agent_test['name']} error: {str(e)}")
            all_passed = False

    return all_passed


def test_frontend_connection(frontend_url: str = "http://localhost:3000") -> bool:
    """Test frontend connection"""
    print_header("Test 5: Frontend Connection")

    try:
        # Test frontend health
        response = requests.get(f"{frontend_url}/api/plant-ai", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Frontend đang chạy tại {frontend_url}")
            print_info(f"Backend status: {data.get('status', 'unknown')}")
            return True
        else:
            print_warning(f"Frontend trả về status code: {response.status_code}")
            print_info("Frontend có thể chưa start hoặc route chưa đúng")
            return False
    except requests.exceptions.ConnectionError:
        print_warning(f"Không thể kết nối đến {frontend_url}")
        print_info("Frontend có thể chưa start: cd FE/ai-chatbot-main && npm run dev")
        return False
    except Exception as e:
        print_warning(f"Lỗi: {str(e)}")
        return False


def check_environment() -> Dict[str, bool]:
    """Check environment variables"""
    print_header("Test 0: Environment Check")

    import os

    from dotenv import load_dotenv

    load_dotenv()

    checks = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "LLM_PROVIDER": bool(os.getenv("LLM_PROVIDER", "groq")),
    }

    if checks["GROQ_API_KEY"]:
        print_success("GROQ_API_KEY đã được set")
    else:
        print_warning("GROQ_API_KEY chưa được set - LLM features sẽ bị disable")
        print_info("Tạo file .env với: GROQ_API_KEY=your_key_here")

    if checks["OPENAI_API_KEY"]:
        print_success("OPENAI_API_KEY đã được set")
    else:
        print_info("OPENAI_API_KEY chưa được set (không bắt buộc nếu dùng Groq)")

    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    print_info(f"LLM Provider: {llm_provider}")

    return checks


def main():
    """Main test function"""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}{'KIỂM TRA TOÀN BỘ FLOW HỆ THỐNG':^60}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    results = {}

    # Test 0: Environment
    env_checks = check_environment()
    results["environment"] = any(env_checks.values())

    # Test 1: Backend Health
    results["backend_health"] = test_backend_health()

    if not results["backend_health"]:
        print_error("\n❌ Backend không chạy! Vui lòng start backend trước:")
        print_info("   python api_server.py")
        sys.exit(1)

    # Test 2: API Root
    results["api_root"] = test_api_root()

    # Test 3: Chat Endpoint (Full Flow)
    results["chat_endpoint"] = test_chat_endpoint()

    # Test 4: Individual Agents
    results["agents"] = test_agents_individually()

    # Test 5: Frontend Connection (optional)
    results["frontend"] = test_frontend_connection()

    # Summary
    print_header("TÓM TẮT KẾT QUẢ")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Kết quả: {passed_tests}/{total_tests} tests passed{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    if passed_tests == total_tests:
        print_success("🎉 Tất cả tests đều PASSED! Hệ thống hoạt động tốt.")
        return 0
    else:
        print_warning("⚠️  Một số tests FAILED. Vui lòng kiểm tra lại.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test bị hủy bởi user{RESET}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Lỗi không mong đợi: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

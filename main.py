"""
Main entry point cho Multi-Agent System
"""

import asyncio
import json

import config

from orchestrator import AgentOrchestrator


async def main():
    """Hàm main để chạy hệ thống"""
    print("=" * 60)
    print("HỆ THỐNG MULTI-AGENT TƯ VẤN")
    print("=" * 60)
    print()

    # Kiểm tra API key
    if not config.OPENAI_API_KEY:
        print("⚠️  Cảnh báo: Chưa cấu hình OPENAI_API_KEY trong file .env")
        print("   Một số tính năng có thể không hoạt động đầy đủ.")
        print()

    # Khởi tạo orchestrator
    orchestrator = AgentOrchestrator()

    # Hiển thị trạng thái agents
    print("Trạng thái các agents:")
    status = orchestrator.get_agent_status()
    for agent_id, agent_status in status.items():
        print(f"  - {agent_status['name']}: {agent_status['status']}")
    print()

    # Ví dụ sử dụng
    print("Ví dụ sử dụng hệ thống nhận dạng và tư vấn bệnh cây trồng:")
    print()

    # Ví dụ 1: Câu hỏi về bệnh cây trồng
    example1 = {
        "user_query": "Cây cà chua của tôi có lá bị vàng và đốm nâu, xin tư vấn cách điều trị",
        "user_context": {"plant_type": "cà chua", "location": "miền Bắc", "season": "mùa mưa"},
    }

    print("Ví dụ 1: Câu hỏi đơn giản")
    print(f"Query: {example1['user_query']}")
    print()

    result1 = await orchestrator.execute(example1)
    print("\n" + "=" * 60)
    print("KẾT QUẢ:")
    print("=" * 60)
    print(json.dumps(result1.get("final_advice", {}), indent=2, ensure_ascii=False))
    print()

    # Ví dụ 2: Với hình ảnh (nếu có)
    # example2 = {
    #     "user_query": "Phân tích hình ảnh này cho tôi",
    #     "image_path": "data/images/sample.jpg"
    # }
    # result2 = await orchestrator.execute(example2)

    # Ví dụ 3: Với dataset (nếu có)
    # example3 = {
    #     "user_query": "Phân tích dataset này",
    #     "dataset_path": "data/datasets/sample.csv"
    # }
    # result3 = await orchestrator.execute(example3)


async def interactive_mode():
    """Chế độ tương tác với người dùng"""
    print("=" * 60)
    print("CHẾ ĐỘ TƯƠNG TÁC")
    print("=" * 60)
    print()

    orchestrator = AgentOrchestrator()

    while True:
        try:
            print("\nNhập câu hỏi của bạn (hoặc 'quit' để thoát):")
            user_query = input("> ").strip()

            if user_query.lower() in ["quit", "exit", "q"]:
                print("Cảm ơn bạn đã sử dụng hệ thống!")
                break

            if not user_query:
                continue

            # Hỏi thêm thông tin
            print("\nBạn có muốn cung cấp thêm thông tin không?")
            print("1. Hình ảnh (nhập đường dẫn)")
            print("2. Dataset (nhập đường dẫn)")
            print("3. Bỏ qua")

            choice = input("Lựa chọn (1/2/3): ").strip()

            user_input = {"user_query": user_query, "user_context": {}}

            if choice == "1":
                image_path = input("Đường dẫn hình ảnh: ").strip()
                if image_path:
                    user_input["image_path"] = image_path

            elif choice == "2":
                dataset_path = input("Đường dẫn dataset: ").strip()
                if dataset_path:
                    user_input["dataset_path"] = dataset_path

            # Thực thi
            print("\nĐang xử lý...")
            result = await orchestrator.execute(user_input)

            # Hiển thị kết quả
            print("\n" + "=" * 60)
            print("KẾT QUẢ TƯ VẤN:")
            print("=" * 60)

            final_advice = result.get("final_advice", {})
            if final_advice:
                print("\nTư vấn:")
                print(
                    final_advice.get("full_advice", final_advice.get("summary", "Không có tư vấn"))
                )

                if final_advice.get("recommendations"):
                    print("\nKhuyến nghị:")
                    for i, rec in enumerate(final_advice["recommendations"], 1):
                        print(f"  {i}. {rec}")

                if final_advice.get("next_steps"):
                    print("\nBước tiếp theo:")
                    for i, step in enumerate(final_advice["next_steps"], 1):
                        print(f"  {i}. {step}")

            print(f"\nĐộ tin cậy: {result.get('final_advice', {}).get('confidence_score', 0):.2%}")

        except KeyboardInterrupt:
            print("\n\nCảm ơn bạn đã sử dụng hệ thống!")
            break
        except Exception as e:
            print(f"\nLỗi: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())

"""
Agent 4: Tìm kiếm & tóm tắt thông tin bệnh cây trồng từ mạng xã hội
"""

import asyncio
from typing import Any, Dict, List

import aiohttp
import config
import requests
from bs4 import BeautifulSoup

from agents.base_agent import BaseAgent


class SocialMediaSearchAgent(BaseAgent):
    """Agent tìm kiếm và tóm tắt thông tin về bệnh cây trồng từ mạng xã hội"""

    def __init__(self):
        super().__init__("agent4", config.AGENT_CONFIG["agent4"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tìm kiếm và tóm tắt thông tin bệnh cây trồng từ mạng xã hội
        """
        search_query = input_data.get("search_query", "")
        keywords = input_data.get("keywords", [])
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})

        if not search_query and not keywords:
            # Sử dụng user_query nếu không có search_query
            search_query = user_query

        # Tìm kiếm từ các nguồn
        search_results = await self._search_social_media(search_query, keywords)

        # Tóm tắt kết quả
        summary = await self._summarize_results(search_results, user_query, context)

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": {
                "search_query": search_query,
                "results_count": len(search_results),
                "search_results": search_results,
                "summary": summary,
            },
            "next_agents": ["agent5"],  # Luôn chuyển đến agent tổng hợp
        }

    async def _search_social_media(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Tìm kiếm từ các nguồn mạng xã hội"""
        results = []

        # Tìm kiếm từ các nguồn khác nhau (mô phỏng)
        # Trong thực tế, có thể tích hợp với Twitter API, Facebook API, Reddit API, etc.

        # 1. Tìm kiếm web (mô phỏng tìm kiếm từ mạng xã hội)
        web_results = await self._search_web(query, keywords)
        results.extend(web_results)

        # 2. Tìm kiếm từ các forum/diễn đàn
        forum_results = await self._search_forums(query, keywords)
        results.extend(forum_results)

        # Giới hạn số lượng kết quả
        return results[:20]

    async def _search_web(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Tìm kiếm từ web (mô phỏng tìm kiếm mạng xã hội)"""
        results = []

        try:
            # Mô phỏng tìm kiếm - trong thực tế sẽ gọi API thật
            # Ví dụ: Google Search API, Twitter API, Facebook Graph API

            # Tạo kết quả mẫu dựa trên query về bệnh cây trồng
            # Trong thực tế, sẽ tích hợp với API thật (Facebook, Twitter, Reddit, diễn đàn nông nghiệp)
            plant_disease_sources = [
                "Diễn đàn nông nghiệp",
                "Nhóm Facebook nông dân",
                "Cộng đồng trồng trọt",
                "Forum cây trồng",
            ]

            for i in range(3):
                results.append(
                    {
                        "source": (
                            plant_disease_sources[i]
                            if i < len(plant_disease_sources)
                            else "social_media"
                        ),
                        "title": f"Thảo luận về bệnh cây: '{query}' - {i+1}",
                        "content": f"Người dùng chia sẻ kinh nghiệm về {query} và các từ khóa {', '.join(keywords[:3])}. Thông tin về triệu chứng, cách điều trị và phòng ngừa.",
                        "url": f"https://example.com/result/{i+1}",
                        "relevance_score": 0.8 - (i * 0.1),
                        "timestamp": "2024-01-01",
                        "author": f"Nông dân {i+1}",
                        "likes": 10 + i * 5,
                        "comments": 3 + i,
                    }
                )
        except Exception as e:
            print(f"Error in web search: {e}")

        return results

    async def _search_forums(self, query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Tìm kiếm từ các forum/diễn đàn"""
        results = []

        try:
            # Mô phỏng tìm kiếm từ forum nông nghiệp
            forum_types = ["Diễn đàn nông nghiệp Việt Nam", "Cộng đồng trồng trọt"]

            for i in range(2):
                results.append(
                    {
                        "source": forum_types[i] if i < len(forum_types) else "forum",
                        "title": f"Thảo luận về bệnh cây '{query}' - {i+1}",
                        "content": f"Thảo luận chi tiết về {query} với các chủ đề: triệu chứng, nguyên nhân, cách điều trị, kinh nghiệm thực tế từ nông dân",
                        "url": f"https://forum.example.com/thread/{i+1}",
                        "relevance_score": 0.7 - (i * 0.1),
                        "timestamp": "2024-01-01",
                        "author": f"Nông dân {i+1}",
                        "replies_count": 10 + i,
                        "views": 100 + i * 50,
                    }
                )
        except Exception as e:
            print(f"Error in forum search: {e}")

        return results

    async def _summarize_results(
        self, results: List[Dict[str, Any]], user_query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tóm tắt kết quả tìm kiếm sử dụng LLM"""
        if not results:
            return {
                "summary": "Không tìm thấy kết quả phù hợp",
                "key_points": [],
                "sentiment": "neutral",
            }

        if not self.client:
            # Tóm tắt đơn giản không dùng LLM
            return self._simple_summarize(results)

        try:
            # Tạo nội dung từ kết quả
            results_text = "\n\n".join(
                [
                    f"Kết quả {i+1}:\nTiêu đề: {r.get('title', '')}\nNội dung: {r.get('content', '')[:200]}"
                    for i, r in enumerate(results[:10])
                ]
            )

            prompt = f"""
            Bạn là chuyên gia nông nghiệp. Tóm tắt và phân tích các kết quả tìm kiếm về bệnh cây trồng từ mạng xã hội sau:

            Câu hỏi của người dùng: {user_query}
            Ngữ cảnh: {context}

            Kết quả tìm kiếm từ cộng đồng nông dân và diễn đàn:
            {results_text}

            Hãy cung cấp:
            1. Tóm tắt tổng quan về các kết quả:
               - Các vấn đề bệnh cây được đề cập nhiều nhất
               - Kinh nghiệm thực tế từ nông dân
               - Các giải pháp được chia sẻ

            2. Các điểm chính được đề cập:
               - Triệu chứng bệnh phổ biến
               - Nguyên nhân thường gặp
               - Cách điều trị hiệu quả
               - Biện pháp phòng ngừa

            3. Xu hướng/sentiment chung:
               - Mức độ quan tâm của cộng đồng
               - Đánh giá về mức độ nghiêm trọng
               - Hiệu quả của các giải pháp được chia sẻ

            4. Thông tin quan trọng nhất:
               - Giải pháp được đánh giá cao nhất
               - Cảnh báo từ cộng đồng
               - Lưu ý quan trọng

            5. Khuyến nghị dựa trên thông tin tìm được:
               - Tổng hợp kinh nghiệm từ cộng đồng
               - Đề xuất cách tiếp cận tốt nhất
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia nông nghiệp và phân tích thông tin từ cộng đồng nông dân. Bạn có khả năng tóm tắt và phân tích thông tin về bệnh cây trồng từ các nguồn mạng xã hội.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=1500,
            )

            summary_text = response.choices[0].message.content

            # Trích xuất thông tin có cấu trúc
            structured_summary = self._extract_structured_summary(summary_text, results)

            return structured_summary

        except Exception as e:
            print(f"Error in LLM summarization: {e}")
            return self._simple_summarize(results)

    def _simple_summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tóm tắt đơn giản không dùng LLM"""
        return {
            "summary": f"Tìm thấy {len(results)} kết quả liên quan",
            "key_points": [r.get("title", "")[:100] for r in results[:5]],
            "sentiment": "neutral",
            "sources": list(set([r.get("source", "unknown") for r in results])),
        }

    def _extract_structured_summary(
        self, summary_text: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trích xuất thông tin có cấu trúc từ summary"""
        key_points = []
        sentiment = "neutral"

        # Phân tích sentiment đơn giản
        summary_lower = summary_text.lower()
        if any(word in summary_lower for word in ["tích cực", "tốt", "tích cực", "positive"]):
            sentiment = "positive"
        elif any(word in summary_lower for word in ["tiêu cực", "xấu", "negative"]):
            sentiment = "negative"

        # Trích xuất key points
        sentences = summary_text.split(".")
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Câu có ý nghĩa
                key_points.append(sentence.strip())

        return {
            "summary": summary_text,
            "key_points": key_points[:5],
            "sentiment": sentiment,
            "sources": list(set([r.get("source", "unknown") for r in results])),
            "total_results": len(results),
        }

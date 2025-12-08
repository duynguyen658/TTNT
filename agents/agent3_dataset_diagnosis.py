"""
Agent 3: Chẩn đoán bệnh cây trồng dựa trên dataset
"""
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import config
import os
import pandas as pd
import numpy as np


class DatasetDiagnosisAgent(BaseAgent):
    """Agent chẩn đoán bệnh cây trồng dựa trên dataset"""
    
    def __init__(self):
        super().__init__("agent3", config.AGENT_CONFIG["agent3"])
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chẩn đoán bệnh cây trồng dựa trên dataset
        """
        dataset_path = input_data.get("dataset_path")
        dataset_data = input_data.get("dataset_data")  # DataFrame hoặc dict
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})
        
        # Load dataset
        df = None
        if dataset_path and os.path.exists(dataset_path):
            df = self._load_dataset(dataset_path)
        elif dataset_data:
            df = self._convert_to_dataframe(dataset_data)
        
        if df is None or df.empty:
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": "Không tìm thấy dataset để phân tích",
                "output": {}
            }
        
        # Phân tích dataset
        analysis_result = await self._analyze_dataset(df, user_query, context)
        
        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "output": analysis_result,
            "next_agents": ["agent5"]  # Luôn chuyển đến agent tổng hợp
        }
    
    def _load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load dataset từ file"""
        try:
            if file_path.endswith('.csv'):
                # Thử với encoding UTF-8-sig trước (cho tiếng Việt)
                try:
                    return pd.read_csv(file_path, encoding='utf-8-sig')
                except:
                    return pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def _convert_to_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        """Chuyển đổi dữ liệu thành DataFrame"""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return None
        except Exception as e:
            print(f"Error converting to DataFrame: {e}")
            return None
    
    async def _analyze_dataset(self, df: pd.DataFrame, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích dataset"""
        # Thống kê cơ bản
        basic_stats = self._calculate_basic_stats(df)
        
        # Phân tích nâng cao
        advanced_analysis = self._perform_advanced_analysis(df, query)
        
        # Sử dụng LLM để tổng hợp và chẩn đoán
        llm_analysis = await self._llm_analysis(df, basic_stats, advanced_analysis, query, context)
        
        return {
            "basic_statistics": basic_stats,
            "advanced_analysis": advanced_analysis,
            "llm_diagnosis": llm_analysis,
            "dataset_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            }
        }
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tính toán thống kê cơ bản"""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {}
        }
        
        # Thống kê cho các cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def _perform_advanced_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Thực hiện phân tích nâng cao cho dataset bệnh cây trồng"""
        analysis = {
            "correlations": {},
            "patterns": [],
            "anomalies": [],
            "disease_statistics": {},
            "plant_statistics": {}
        }
        
        # Tính correlation cho các cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            analysis["correlations"] = df[numeric_cols].corr().to_dict()
        
        # Phân tích các cột liên quan đến bệnh cây
        disease_keywords = ["bệnh", "disease", "symptom", "triệu chứng", "tình trạng"]
        plant_keywords = ["cây", "plant", "loại", "type", "variety", "giống"]
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Phân tích phân phối bệnh
            if any(keyword in col_lower for keyword in disease_keywords):
                if df[col].dtype in ['object', 'string']:
                    value_counts = df[col].value_counts()
                    analysis["disease_statistics"][col] = {
                        "distribution": value_counts.to_dict(),
                        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                        "count": len(value_counts)
                    }
            
            # Phân tích phân phối cây trồng
            if any(keyword in col_lower for keyword in plant_keywords):
                if df[col].dtype in ['object', 'string']:
                    value_counts = df[col].value_counts()
                    analysis["plant_statistics"][col] = {
                        "distribution": value_counts.to_dict(),
                        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                        "count": len(value_counts)
                    }
            
            # Phát hiện pattern chung
            if df[col].dtype in ['object', 'string']:
                value_counts = df[col].value_counts()
                if len(value_counts) < 20:  # Nếu có ít giá trị unique
                    analysis["patterns"].append({
                        "column": col,
                        "distribution": value_counts.to_dict(),
                        "top_values": value_counts.head(5).to_dict()
                    })
        
        # Phát hiện anomalies (giá trị bất thường)
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    anomalies = df[df[col].abs() > mean_val + 2 * std_val]
                    if len(anomalies) > 0:
                        analysis["anomalies"].append({
                            "column": col,
                            "count": len(anomalies),
                            "percentage": len(anomalies) / len(df) * 100
                        })
        
        return analysis
    
    async def _llm_analysis(self, df: pd.DataFrame, basic_stats: Dict, 
                           advanced_analysis: Dict, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sử dụng LLM để phân tích và chẩn đoán"""
        if not self.client:
            return {
                "diagnosis": "Không thể phân tích: Thiếu API key",
                "insights": []
            }
        
        try:
            # Tạo summary của dataset
            dataset_summary = f"""
            Dataset có {basic_stats['row_count']} hàng và {basic_stats['column_count']} cột.
            Các cột: {', '.join(df.columns.tolist()[:10])}
            Thống kê cơ bản: {str(basic_stats['numeric_summary'])[:500]}
            """
            
            prompt = f"""
            Bạn là chuyên gia nông nghiệp và phân tích dữ liệu bệnh cây trồng. Phân tích dataset sau và đưa ra chẩn đoán/tư vấn:
            
            Thông tin dataset:
            {dataset_summary}
            
            Phân tích nâng cao:
            - Thống kê bệnh: {str(advanced_analysis.get('disease_statistics', {}))[:500]}
            - Thống kê cây trồng: {str(advanced_analysis.get('plant_statistics', {}))[:500]}
            - Patterns: {str(advanced_analysis.get('patterns', []))[:500]}
            
            Câu hỏi của người dùng: {query}
            Ngữ cảnh: {context}
            
            Hãy cung cấp:
            1. Phân tích tổng quan về dataset bệnh cây trồng:
               - Số lượng mẫu bệnh
               - Các loại bệnh phổ biến
               - Các loại cây trồng bị ảnh hưởng
               - Phân bố theo thời gian/địa điểm (nếu có)
            
            2. Các phát hiện quan trọng:
               - Bệnh nào xuất hiện nhiều nhất
               - Loại cây nào dễ bị bệnh nhất
               - Mối tương quan giữa các yếu tố
               - Xu hướng và patterns
            
            3. Chẩn đoán/đánh giá dựa trên dữ liệu:
               - Đánh giá tình trạng bệnh tổng thể
               - Xác định các bệnh nguy hiểm cần chú ý
               - Phân tích nguyên nhân có thể
            
            4. Insights và khuyến nghị:
               - Khuyến nghị phòng ngừa
               - Biện pháp xử lý dựa trên dữ liệu
               - Cảnh báo về các bệnh có nguy cơ cao
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia nông nghiệp và phân tích dữ liệu bệnh cây trồng với nhiều năm kinh nghiệm. Bạn có khả năng phân tích dataset và đưa ra insights hữu ích về bệnh cây trồng."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "raw_analysis": analysis_text,
                "diagnosis": analysis_text[:500],
                "insights": self._extract_insights(analysis_text),
                "recommendations": self._extract_recommendations(analysis_text)
            }
            
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {
                "error": str(e),
                "diagnosis": "Lỗi trong quá trình phân tích dataset",
                "insights": []
            }
    
    def _extract_insights(self, analysis_text: str) -> list:
        """Trích xuất insights từ kết quả phân tích"""
        # Logic đơn giản - trong thực tế có thể dùng LLM hoặc NLP
        insights = []
        sentences = analysis_text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["quan trọng", "đáng chú ý", "phát hiện", "insight"]):
                insights.append(sentence.strip())
        return insights[:5]  # Giới hạn 5 insights
    
    def _extract_recommendations(self, analysis_text: str) -> list:
        """Trích xuất recommendations từ kết quả phân tích"""
        recommendations = []
        sentences = analysis_text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["nên", "khuyến nghị", "recommend", "nên làm"]):
                recommendations.append(sentence.strip())
        return recommendations[:5]  # Giới hạn 5 recommendations


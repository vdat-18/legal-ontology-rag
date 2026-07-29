"""Streamlit application for Corrective RAG (CRAG) visualization.

Provides an interactive user interface to run queries, visualize the retrieval and grading
phases, and inspect document reference citations.
"""

from __future__ import annotations

import streamlit as st

from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline

# Page configuration
st.set_page_config(page_title="Corrective RAG - Luật Giao Thông Việt Nam", page_icon="⚖️", layout="wide")

# App Custom CSS styling for premium look and feel
st.markdown(
    """
<style>
    /* Dark mode styling & modern font styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header card styled with premium gradient background */
    .header-container {
        padding: 2.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        text-align: center;
    }
    .header-title {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Answer box layout */
    .answer-container {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 4px 12px 12px 4px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border: 1px solid #334155;
    }
    
    /* Document citation cards */
    .chunk-card {
        background-color: #1a1f2c;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #2e374a;
    }
    .chunk-card-relevant {
        border-left: 4px solid #10b981;
        background-color: rgba(16, 185, 129, 0.05);
    }
    .chunk-card-irrelevant {
        border-left: 4px solid #ef4444;
        background-color: rgba(239, 68, 68, 0.05);
        opacity: 0.7;
    }
    .badge-relevant {
        background-color: #064e3b;
        color: #34d399;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-irrelevant {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .source-title {
        font-weight: bold;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_pipeline() -> CorrectiveRAGPipeline | None:
    """Load configuration and initialize pipeline."""
    try:
        config = RAGConfig.from_env()
        # Cache pipeline to avoid loading embedding models on every rerun
        if "pipeline" not in st.session_state:
            with st.spinner("Đang tải mô hình ngôn ngữ và sinh vector (SentenceTransformer)..."):
                st.session_state.pipeline = CorrectiveRAGPipeline(config)
    except Exception as e:
        st.error(f"Lỗi khởi tạo hệ thống RAG: {e}")
        st.info("Vui lòng đảm bảo bạn đã tạo file `.env` từ `.env.example` và thiết lập các biến môi trường.")
        return None
    else:
        return st.session_state.pipeline


def main() -> None:
    """Orchestrate Streamlit web page logic."""
    # Header Section
    st.markdown(
        """
    <div class="header-container">
        <div class="header-title">⚖️ Corrective RAG (CRAG)</div>
        <div class="header-subtitle">Hệ thống hỏi đáp Luật Giao thông đường bộ Việt Nam được kiểm chứng thông tin chính xác</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize RAG Pipeline
    pipeline = init_pipeline()
    if pipeline is None:
        return

    # Sidebar parameters
    st.sidebar.title("Cấu hình RAG")
    st.sidebar.markdown(f"**Embedding Profile:** `{pipeline.config.embedding_profile}`")
    st.sidebar.markdown(f"**Mô hình LLM:** `{pipeline.config.llm_model}`")
    st.sidebar.markdown("---")

    limit = st.sidebar.slider("Số lượng đoạn trích lấy ra (K)", min_value=2, max_value=10, value=2, step=1)

    # Search Bar
    query = st.text_input(
        "Nhập câu hỏi của bạn về luật giao thông đường bộ:",
        placeholder="Ví dụ: Vượt đèn đỏ xe máy bị phạt bao nhiêu tiền và có bị trừ điểm bằng lái không?",
    )

    if query:
        # Run query through RAG
        with st.status("Đang chạy luồng Corrective RAG...", expanded=True) as status:
            st.write("🔍 Đang tìm kiếm đoạn trích trong cơ sở dữ liệu Milvus...")

            # Run pipeline
            result = pipeline.run(query, limit=limit)

            st.write("⚖️ Đang chấm điểm và thẩm định mức độ liên quan của tài liệu...")

            if result["reformulated_query"]:
                st.write(
                    f"🔄 Không tìm thấy tài liệu phù hợp trực tiếp. Đang tối ưu hóa truy vấn thành: *{result['reformulated_query']}*..."
                )
                st.write("🔍 Đang tìm kiếm lại với truy vấn mới...")

            st.write("✍️ Đang tổng hợp câu trả lời dựa trên tài liệu tham chiếu đã chọn lọc...")
            status.update(label="Hoàn tất quy trình xử lý!", state="complete", expanded=False)

        # Display results
        st.markdown("### 📝 Câu trả lời:")
        st.markdown(f'<div class="answer-container">{result["answer"]}</div>', unsafe_allow_html=True)

        st.subheader("Báo cáo luồng xử lý RAG")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trạng thái Pipeline", result["status"])
        with col2:
            st.metric("CRAG Action", result["crag_action"])
        with col3:
            st.metric("Internal Knowledge", result["internal_knowledge_count"])
        with col4:
            st.metric("External Knowledge", result["external_knowledge_count"])

        # Display document details (Graded Chunks)
        st.markdown("### 📚 Chi tiết đoạn trích tham chiếu và Thẩm định:")

        for item in result["graded_chunks"]:
            is_relevant = item["is_relevant"]
            card_class = "chunk-card-relevant" if is_relevant else "chunk-card-irrelevant"
            badge = (
                f'<span class="badge-relevant">{item["action"]}</span>'
                if is_relevant
                else f'<span class="badge-irrelevant">{item["action"]}</span>'
            )

            second_attempt_label = "*(Tìm kiếm lại)* " if item.get("is_second_attempt") else ""

            st.markdown(
                f"""
            <div class="chunk-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="source-title">📄 {second_attempt_label}{item["source_name"]} (Similarity: {item["score"]:.4f}; Evaluator: {item["confidence_score"]:.2f})</div>
                    <div>{badge}</div>
                </div>
                <div style="font-size: 0.95rem; white-space: pre-wrap; color: #cbd5e1;">{item["text"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

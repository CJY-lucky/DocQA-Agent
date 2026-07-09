"""混合检索引擎 —— 融合向量检索（语义）和 BM25 检索（关键词）的结果。"""

from __future__ import annotations

from src import logger
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import RetrievalResult, VectorStore


class HybridRetriever:
    """混合检索器，使用 RRF（Reciprocal Rank Fusion）融合两种检索结果。"""

    # RRF 常数，控制排名融合的平滑度
    RRF_K = 60

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Args:
            vector_store: 向量存储实例
            bm25_store: BM25 存储实例
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """执行混合检索并融合结果。

        Args:
            query: 查询文本
            top_k: 最终返回的结果数量

        Returns:
            融合后的 RetrievalResult 列表
        """
        # 每个源检索更多的结果，以便融合后有足够候选
        fetch_k = max(top_k * 3, 15)

        vector_results = self.vector_store.search(query, top_k=fetch_k)
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        if not vector_results and not bm25_results:
            return []

        # RRF 融合：对每个唯一内容计算加权 RRF 分数
        rrf_scores: dict[str, tuple[float, RetrievalResult]] = {}

        def _add_results(results: list[RetrievalResult], weight: float):
            for rank, result in enumerate(results, start=1):
                rrf = weight / (self.RRF_K + rank)
                key = f"{result.file_path}:{result.chunk_index}"
                if key in rrf_scores:
                    prev_score, _ = rrf_scores[key]
                    rrf_scores[key] = (prev_score + rrf, result)
                else:
                    rrf_scores[key] = (rrf, result)

        _add_results(vector_results, self.vector_weight)
        _add_results(bm25_results, self.bm25_weight)

        # 按融合分数排序
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        top_items = sorted_items[:top_k]

        # 归一化最终分数到 [0, 1]
        max_score = top_items[0][0] if top_items else 1.0

        merged: list[RetrievalResult] = []
        for fused_score, result in top_items:
            result.score = round(fused_score / max_score, 4) if max_score > 0 else 0.0
            result.source = "hybrid"
            merged.append(result)

        logger.info(
            "混合检索: 向量=%d, BM25=%d → 融合=%d",
            len(vector_results), len(bm25_results), len(merged),
        )
        return merged

    def index_chunks(self, chunks) -> tuple[int, int]:
        """同时索引到两种存储。"""
        v_count = self.vector_store.index_chunks(chunks)
        b_count = self.bm25_store.index_chunks(chunks)
        return v_count, b_count

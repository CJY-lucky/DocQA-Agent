"""BM25 存储 —— 基于关键词的稀疏检索。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src import logger
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import RetrievalResult


def _tokenize(text: str) -> list[str]:
    """简易中文+英文分词。"""
    # 中文字符间加空格，英文按空白分割
    text = re.sub(r"([一-鿿])", r" \1 ", text)
    tokens = text.lower().split()
    # 过滤过短的 token
    return [t for t in tokens if len(t) > 0]


class BM25Store:
    """BM25 关键词检索引擎。"""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """建立 BM25 索引。

        Args:
            chunks: 文本块列表

        Returns:
            索引的块数量
        """
        self._chunks = list(chunks)
        self._tokenized = [_tokenize(c.content) for c in chunks]

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

        logger.info("BM25 索引完成: %d 个块", len(chunks))
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """BM25 关键词检索。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            RetrievalResult 列表（按 BM25 分数降序）
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 索引为空")
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        # 归一化分数到 [0, 1]
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = scores[idx]
            # Min-Max 归一化
            if max_score > min_score:
                normalized = (float(score) - min_score) / (max_score - min_score)
            else:
                normalized = 0.5

            chunk = self._chunks[idx]
            results.append(RetrievalResult(
                content=chunk.content,
                score=round(normalized, 4),
                file_path=chunk.file_path,
                file_name=chunk.file_name,
                chunk_index=chunk.chunk_index,
                source="bm25",
            ))

        return results

    @property
    def count(self) -> int:
        """已索引的块数量。"""
        return len(self._chunks)

"""向量存储 —— 基于 FAISS + numpy 的语义检索。

设计：用 Python 的 numpy 保存原始嵌入向量 → 启动时重建 FAISS 索引。
完全绕开 FAISS C++ 层的文件 I/O（Windows 中文路径和 API 兼容性问题）。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import logger
from src.ingestion.chunker import Chunk


@dataclass
class RetrievalResult:
    """检索结果。"""

    content: str
    score: float  # 0-1 之间，越高越相关
    file_path: str
    file_name: str
    chunk_index: int
    source: str  # "vector" | "bm25"


class VectorStore:
    """FAISS 向量存储。数据用 numpy (.npy) + pickle (.pkl) 持久化。"""

    def __init__(
        self,
        persist_dir: str = "./data/faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self._embeddings_path = os.path.join(self.persist_dir, "embeddings.npy")
        self._docs_path = os.path.join(self.persist_dir, "docs.pkl")

        logger.info("加载嵌入模型: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model, device=device)
        self._embed_dim = self._embedder.get_sentence_embedding_dimension()

        self._index: faiss.Index | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._cached_count: int = self._load_existing()

    # ------------------------------------------------------------------
    # 持久化（纯 Python I/O，不经过 FAISS C++ 文件层）
    # ------------------------------------------------------------------

    def _load_existing(self) -> int:
        """从 numpy + pickle 加载数据，重建 FAISS 索引。"""
        if os.path.exists(self._embeddings_path) and os.path.exists(self._docs_path):
            try:
                embeddings = np.load(self._embeddings_path)
                with open(self._docs_path, "rb") as f:
                    data = pickle.load(f)
                    self._documents = data.get("documents", [])
                    self._metadatas = data.get("metadatas", [])

                # 从 numpy 数组重建 FAISS 索引（纯内存操作）
                self._index = faiss.IndexFlatIP(self._embed_dim)
                self._index.add(embeddings)

                count = len(self._documents)
                logger.info("从磁盘加载了 %d 个向量", count)
                return count
            except Exception as e:
                logger.warning("加载已有索引失败: %s", e)
                self._index = None
                self._documents = []
                self._metadatas = []
        return 0

    def _save(self, embeddings: np.ndarray, count: int) -> None:
        """保存嵌入向量（numpy）和文档（pickle）。纯 Python I/O。"""
        os.makedirs(self.persist_dir, exist_ok=True)
        np.save(self._embeddings_path, embeddings)
        with open(self._docs_path, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "metadatas": self._metadatas,
            }, f)
        logger.info("数据已持久化: %s (%d 个向量)", self.persist_dir, count)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """将分块嵌入并建立 FAISS 索引。"""
        if not chunks:
            logger.warning("没有可分块的文档")
            return 0

        texts = [c.content for c in chunks]

        logger.info("正在为 %d 个块生成嵌入向量...", len(texts))
        embeddings = self._embedder.encode(texts, show_progress_bar=True).astype(np.float32)

        # L2 归一化 → 内积 = 余弦相似度
        faiss.normalize_L2(embeddings)

        # 构建 FAISS 索引（纯内存，不涉及文件）
        self._index = faiss.IndexFlatIP(self._embed_dim)
        self._index.add(embeddings)

        # 文档和元数据
        self._documents = texts
        import hashlib
        self._metadatas = [
            {
                "file_path": c.file_path,
                "file_name": c.file_name,
                "file_type": c.file_type,
                "chunk_index": c.chunk_index,
                "id": f"{hashlib.md5(c.file_path.encode()).hexdigest()[:8]}_{c.file_name}_{c.chunk_index}",
            }
            for c in chunks
        ]

        # 持久化（numpy + pickle）
        self._save(embeddings, len(chunks))
        self._cached_count = len(chunks)
        logger.info("FAISS 索引完成: %d 个块", len(chunks))
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """语义检索。"""
        if self._index is None or not self._documents:
            return []

        query_embedding = self._embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(query_embedding, k)

        results: list[RetrievalResult] = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self._documents):
                continue

            score = max(0.0, (float(distances[0][i]) + 1.0) / 2.0)

            meta = self._metadatas[idx] if idx < len(self._metadatas) else {}
            results.append(RetrievalResult(
                content=self._documents[idx],
                score=round(score, 4),
                file_path=meta.get("file_path", ""),
                file_name=meta.get("file_name", ""),
                chunk_index=meta.get("chunk_index", 0),
                source="vector",
            ))

        return results

    def get_all_chunks(self) -> list[Chunk]:
        """读取所有已索引的文档块（用于重建 BM25 索引）。"""
        chunks: list[Chunk] = []
        for i, doc in enumerate(self._documents):
            meta = self._metadatas[i] if i < len(self._metadatas) else {}
            chunks.append(Chunk(
                content=doc,
                chunk_index=meta.get("chunk_index", 0),
                file_path=meta.get("file_path", ""),
                file_name=meta.get("file_name", ""),
                file_type=meta.get("file_type", ""),
                metadata=meta,
            ))
        if chunks:
            logger.info("从存储读取了 %d 个块", len(chunks))
        return chunks

    def clear(self) -> None:
        """清空所有数据。"""
        self._index = None
        self._documents = []
        self._metadatas = []
        self._cached_count = 0
        for path in (self._embeddings_path, self._docs_path):
            if os.path.exists(path):
                os.remove(path)
        logger.info("向量存储已清空")

    @property
    def count(self) -> int:
        return self._cached_count

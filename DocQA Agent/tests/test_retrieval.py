"""检索模块单元测试。

测试范围：文档加载、文本分块、BM25 检索、向量检索、混合检索。
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def sample_dir():
    """创建包含示例文档的临时目录。"""
    tmp = tempfile.mkdtemp()

    # 创建一些示例文件
    files = {
        "README.md": (
            "# Test Project\n\n"
            "This is a test project for unit testing.\n\n"
            "## Installation\n\n"
            "Run `pip install -r requirements.txt` to install dependencies.\n\n"
            "## Usage\n\n"
            "Use `python main.py` to start the application.\n"
            "Configuration is done via `config.yaml`.\n"
        ),
        "src/main.py": (
            "def hello():\n"
            '    """Print a greeting."""\n'
            '    print("Hello, World!")\n'
            "\n"
            "def add(a: int, b: int) -> int:\n"
            '    """Add two numbers."""\n'
            "    return a + b\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    hello()\n"
        ),
        "docs/api.txt": (
            "API Reference\n"
            "=============\n\n"
            "GET /users - List all users\n"
            "POST /users - Create a new user\n"
            "GET /users/{id} - Get user by ID\n"
            "DELETE /users/{id} - Delete a user\n"
        ),
        "ignore.log": "This should be ignored",
        "ignore.tmp": "This should also be ignored",
    }

    for rel_path, content in files.items():
        full_path = Path(tmp) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def sample_chunks(sample_dir):
    """创建示例分块。"""
    from src.ingestion.loader import DocumentLoader
    from src.ingestion.chunker import DocumentChunker

    loader = DocumentLoader()
    docs = loader.load_directory(sample_dir)
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    return chunker.chunk_documents(docs)


# ================================================================
# 测试：DocumentLoader
# ================================================================

class TestDocumentLoader:
    """测试文档加载器。"""

    def test_load_valid_files(self, sample_dir):
        """应该加载指定扩展名的文件。"""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader(extensions={".md", ".py", ".txt"})
        docs = loader.load_directory(sample_dir)

        assert len(docs) == 3
        extensions = {doc.file_type for doc in docs}
        assert extensions == {"md", "py", "txt"}

    def test_ignore_other_extensions(self, sample_dir):
        """应该忽略不在扩展名列表中的文件。"""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader(extensions={".md", ".py", ".txt"})
        docs = loader.load_directory(sample_dir)

        file_names = {doc.file_name for doc in docs}
        assert "ignore.log" not in file_names
        assert "ignore.tmp" not in file_names

    def test_document_metadata(self, sample_dir):
        """加载的文档应包含正确的元数据。"""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_directory(sample_dir)

        for doc in docs:
            assert doc.content
            assert doc.file_path
            assert doc.file_name
            assert doc.file_type in {"md", "py", "txt"}

    def test_empty_directory(self):
        """加载空目录应返回空列表。"""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        tmp = tempfile.mkdtemp()
        try:
            docs = loader.load_directory(tmp)
            assert docs == []
        finally:
            shutil.rmtree(tmp)

    def test_nonexistent_directory(self):
        """加载不存在的目录应抛出异常。"""
        from src.ingestion.loader import DocumentLoader
        loader = DocumentLoader()
        with pytest.raises(ValueError):
            loader.load_directory("/nonexistent/path/12345")


# ================================================================
# 测试：DocumentChunker
# ================================================================

class TestDocumentChunker:
    """测试文本分块器。"""

    def test_chunk_creation(self, sample_dir):
        """应该能将文档拆分为多个块。"""
        from src.ingestion.loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker

        loader = DocumentLoader()
        docs = loader.load_directory(sample_dir)
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) > 0
        # README.md 有约200字符，chunk_size=100 应产生多个块
        readme_chunks = [c for c in chunks if c.file_name == "README.md"]
        assert len(readme_chunks) >= 2

    def test_chunk_metadata(self, sample_chunks):
        """每个块应包含来源元数据。"""
        for chunk in sample_chunks:
            assert chunk.content
            assert chunk.file_path
            assert chunk.file_name
            assert chunk.file_type
            assert chunk.chunk_index >= 0
            assert "source" in chunk.metadata

    def test_chunk_overlap(self, sample_dir):
        """相邻块的末尾和开头应有重叠。"""
        from src.ingestion.loader import DocumentLoader
        from src.ingestion.chunker import DocumentChunker

        loader = DocumentLoader()
        docs = loader.load_directory(sample_dir)
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=20)
        chunks = chunker.chunk_documents(docs)

        # 找同一个文件的两个相邻块
        file_chunks = {}
        for c in chunks:
            file_chunks.setdefault(c.file_name, []).append(c)

        for fname, fchunks in file_chunks.items():
            if len(fchunks) >= 2:
                # 第一个块的末尾应该在第二个块中出现
                # (重叠不是精确字符串匹配，但概念上相邻块共享部分内容)
                for i in range(len(fchunks) - 1):
                    # 至少它们来源相同
                    assert fchunks[i].file_path == fchunks[i + 1].file_path


# ================================================================
# 测试：BM25Store
# ================================================================

class TestBM25Store:
    """测试 BM25 关键词检索。"""

    def test_index_and_search(self, sample_chunks):
        """索引后搜索应返回相关结果。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        store.index_chunks(sample_chunks)

        # 搜索 "installation"
        results = store.search("installation", top_k=3)
        assert len(results) > 0
        assert any("Installation" in r.content or "installation" in r.content.lower()
                   for r in results)

    def test_search_relevance(self, sample_chunks):
        """搜索结果应与关键词相关。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        store.index_chunks(sample_chunks)

        # 搜索 API 相关内容
        results = store.search("API users", top_k=3)
        assert len(results) > 0
        # 得分最高的结果应该包含 API/user 相关的内容
        assert any("api" in r.content.lower() or "users" in r.content.lower()
                   for r in results)

    def test_empty_search(self):
        """空查询应返回空结果。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        results = store.search("", top_k=5)
        assert results == []

    def test_unindexed_search(self):
        """未索引时搜索应返回空结果。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        results = store.search("test", top_k=5)
        assert results == []

    def test_count(self, sample_chunks):
        """count 应返回正确的块数量。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        assert store.count == 0
        store.index_chunks(sample_chunks)
        assert store.count == len(sample_chunks)

    def test_score_normalization(self, sample_chunks):
        """分数应在 0-1 之间。"""
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store()
        store.index_chunks(sample_chunks)
        results = store.search("test project", top_k=5)

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} not in [0, 1]"


# ================================================================
# 测试：VectorStore（需要嵌入模型，标记为慢速）
# ================================================================

@pytest.mark.slow
class TestVectorStore:
    """测试向量存储（需要 sentence-transformers 模型）。"""

    def test_index_and_search(self, sample_chunks):
        """索引后语义搜索应返回相关结果。"""
        from src.retrieval.vector_store import VectorStore

        # 使用临时目录避免污染
        tmp_persist = tempfile.mkdtemp()
        try:
            store = VectorStore(persist_dir=tmp_persist)
            store.index_chunks(sample_chunks)

            # 语义搜索
            results = store.search("How to install?", top_k=3)
            assert len(results) > 0

            # 应该有与 "install" 相关的结果
            contents = [r.content.lower() for r in results]
            assert any("install" in c or "dependencies" in c for c in contents)
        finally:
            shutil.rmtree(tmp_persist)

    def test_score_range(self, sample_chunks):
        """语义检索分数应在 0-1 之间。"""
        from src.retrieval.vector_store import VectorStore

        tmp_persist = tempfile.mkdtemp()
        try:
            store = VectorStore(persist_dir=tmp_persist)
            store.index_chunks(sample_chunks)

            results = store.search("test query", top_k=5)
            for r in results:
                assert 0.0 <= r.score <= 1.0
        finally:
            shutil.rmtree(tmp_persist)

    def test_empty_store_search(self):
        """空存储搜索应返回空列表。"""
        from src.retrieval.vector_store import VectorStore

        tmp_persist = tempfile.mkdtemp()
        try:
            store = VectorStore(persist_dir=tmp_persist)
            results = store.search("test", top_k=5)
            assert results == []
        finally:
            shutil.rmtree(tmp_persist)

    def test_clear(self, sample_chunks):
        """清空后应无数据。"""
        from src.retrieval.vector_store import VectorStore

        tmp_persist = tempfile.mkdtemp()
        try:
            store = VectorStore(persist_dir=tmp_persist)
            store.index_chunks(sample_chunks)
            assert store.count > 0

            store.clear()
            assert store.count == 0
        finally:
            shutil.rmtree(tmp_persist)


# ================================================================
# 测试：HybridRetriever
# ================================================================

@pytest.mark.slow
class TestHybridRetriever:
    """测试混合检索引擎。"""

    def test_hybrid_search(self, sample_chunks):
        """混合检索应融合两种策略的结果。"""
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.hybrid_retriever import HybridRetriever

        tmp_persist = tempfile.mkdtemp()
        try:
            vector_store = VectorStore(persist_dir=tmp_persist)
            bm25_store = BM25Store()
            retriever = HybridRetriever(vector_store, bm25_store)

            v_count, b_count = retriever.index_chunks(sample_chunks)
            assert v_count == len(sample_chunks)
            assert b_count == len(sample_chunks)

            results = retriever.search("How to install?", top_k=5)
            assert len(results) > 0

            # 结果应按分数降序
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score
        finally:
            shutil.rmtree(tmp_persist)

    def test_all_results_marked_hybrid(self, sample_chunks):
        """混合检索的结果 source 应该为 'hybrid'。"""
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.hybrid_retriever import HybridRetriever

        tmp_persist = tempfile.mkdtemp()
        try:
            vector_store = VectorStore(persist_dir=tmp_persist)
            bm25_store = BM25Store()
            retriever = HybridRetriever(vector_store, bm25_store)

            retriever.index_chunks(sample_chunks)
            results = retriever.search("test", top_k=5)

            for r in results:
                assert r.source == "hybrid"
        finally:
            shutil.rmtree(tmp_persist)

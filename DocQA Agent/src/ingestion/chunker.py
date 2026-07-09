"""文本分块器 —— 使用 LangChain 的 RecursiveCharacterTextSplitter 进行语义分块。"""

from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import logger
from src.ingestion.loader import Document


@dataclass
class Chunk:
    """表示一个文本块。"""

    content: str
    chunk_index: int
    file_path: str
    file_name: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """将 Document 列表分割为适合检索的小块。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        """
        Args:
            chunk_size: 每个分块的目标字符数
            chunk_overlap: 相邻分块之间的重叠字符数
            separators: 分隔符优先级列表（默认按 Markdown/代码段落分）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or [
                "\n## ", "\n# ", "\n### ",
                "\n```\n", "\n\n", "\n",
                "。", ". ", " ",
                "",
            ],
            length_function=len,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """将文档列表分块。

        Args:
            documents: Document 对象列表

        Returns:
            Chunk 对象列表
        """
        chunks: list[Chunk] = []
        for doc in documents:
            texts = self._splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunk = Chunk(
                    content=text.strip(),
                    chunk_index=i,
                    file_path=doc.file_path,
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    metadata={
                        "source": doc.file_path,
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "chunk_index": i,
                        "total_chunks": len(texts),
                    },
                )
                chunks.append(chunk)
        logger.info("将 %d 个文档分块为 %d 个块", len(documents), len(chunks))
        return chunks

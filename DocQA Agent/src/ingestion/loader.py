"""文档加载器 —— 扫描目录，加载指定类型的文件。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src import logger


@dataclass
class Document:
    """表示一个加载的文档。"""

    content: str
    file_path: str
    file_name: str
    file_type: str  # "md" | "txt" | "py"
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """扫描指定目录，加载 .md / .txt / .py 文件为 Document 列表。"""

    DEFAULT_EXTENSIONS = {".md", ".txt", ".py"}

    def __init__(self, extensions: set[str] | None = None):
        """
        Args:
            extensions: 要加载的文件扩展名集合，默认 {".md", ".txt", ".py"}
        """
        self.extensions = extensions or self.DEFAULT_EXTENSIONS

    def load_directory(self, directory: str | Path) -> list[Document]:
        """递归扫描目录，加载所有匹配的文件。

        Args:
            directory: 目标目录路径

        Returns:
            Document 列表
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"路径不存在或不是目录: {directory}")

        documents: list[Document] = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = Path(root) / file_name
                ext = file_path.suffix.lower()
                if ext not in self.extensions:
                    continue

                try:
                    content = self._read_file(file_path)
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning("跳过无法读取的文件 %s: %s", file_path, e)
                    continue

                if not content.strip():
                    continue

                doc = Document(
                    content=content,
                    file_path=str(file_path.resolve()),
                    file_name=file_name,
                    file_type=ext.lstrip("."),
                )
                documents.append(doc)

        logger.info("从 %s 加载了 %d 个文档", directory, len(documents))
        return documents

    def _read_file(self, file_path: Path) -> str:
        """读取文件内容，自动尝试多种编码。"""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"无法用任何编码读取: {file_path}")

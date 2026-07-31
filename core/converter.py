"""文档转换器 - 用 MarkItDown 将 PDF/Word 等文件转为文本"""
from __future__ import annotations

import os
from pathlib import Path

from .models import Document

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".txt", ".md"}


class DocumentConverter:
    """封装 MarkItDown，将各种格式文档转为纯文本"""

    def __init__(self):
        from markitdown import MarkItDown
        self._converter = MarkItDown()

    def convert(self, file_path: str | Path) -> Document:
        """转换单个文件为 Document"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {path.suffix}（支持: {SUPPORTED_EXTENSIONS}）")

        result = self._converter.convert(str(path))
        return Document(
            text=result.text_content or "",
            source=path.name,
        )

    def convert_directory(self, dir_path: str | Path) -> list[Document]:
        """批量转换目录下所有支持的文件"""
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"目录不存在: {directory}")

        documents: list[Document] = []
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    doc = self.convert(path)
                    if doc.text.strip():
                        documents.append(doc)
                except Exception as e:
                    print(f"  [跳过] {path.name}: {e}")

        return documents
from io import BytesIO

from pypdf import PdfReader


class FileTextExtractor:
    def extract_text(
        self,
        filename: str,
        content: bytes,
    ) -> str:
        if filename.endswith(".txt"):
            return self._extract_txt(content)

        if filename.endswith(".pdf"):
            return self._extract_pdf(content)

        raise ValueError("Unsupported file format. Only .txt and .pdf are supported.")

    def _extract_txt(self, content: bytes) -> str:
        return content.decode("utf-8")

    def _extract_pdf(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))

        pages_text = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages_text.append(text)

        return "\n\n".join(pages_text)

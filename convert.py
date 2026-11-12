import pymupdf4llm
from pathlib import Path

pdf_path = Path("data/pp_bak_mag_shgpu_2026.pdf")
md_path = pdf_path.with_suffix(".md")

# Конвертация PDF в Markdown
markdown = pymupdf4llm.to_markdown(
    str(pdf_path),
    page_chunks=False
)

# Сохранение результата
md_path.write_text(markdown, encoding="utf-8")

print(f"Конвертация завершена: {md_path}")
print(f"Размер текста: {len(markdown)} символов")
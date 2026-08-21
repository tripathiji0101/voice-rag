from pathlib import Path


def load_text_file(file_path: str | Path) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return path.read_text(encoding="utf-8")


def load_documents(directory: str | Path) -> list[dict]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(directory.glob("*.txt")):
        text = load_text_file(file_path)

        if not text.strip():
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                },
            }
        )

    return documents

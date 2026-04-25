def cell_text(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().replace("\n", " ").replace("|", "\\|")

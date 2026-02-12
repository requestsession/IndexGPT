def select_compare_sources(hits, max_papers=5):
    """Pick top unique-paper evidence snippets for comparison."""
    if not hits:
        return []

    sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
    picked = []
    seen_sources = set()

    for item in sorted_hits:
        source = item.get("source")
        if not source or source in seen_sources:
            continue
        picked.append(item)
        seen_sources.add(source)
        if len(picked) >= max_papers:
            break

    return picked


def build_compare_query(user_query: str) -> str:
    return (
        f"{user_query}\n\n"
        "Please compare the relevant papers and output:\n"
        "1) A markdown table with columns: Paper | Method | Key Findings | Limitations.\n"
        "2) A short synthesis (3-5 bullet points) after the table.\n"
        "Use only provided excerpts and cite source+page in each row."
    )

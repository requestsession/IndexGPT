const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
  "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
  "when", "where", "which", "who", "why", "with"
]);

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getQueryTerms(query) {
  const rawTerms = query
    .toLowerCase()
    .match(/[\p{L}\p{N}_-]{2,}/gu) || [];

  const deduped = [...new Set(rawTerms.filter((term) => !STOPWORDS.has(term)))];
  return deduped.sort((a, b) => b.length - a.length);
}

export function splitHighlightedText(text, query) {
  if (!text) return [];

  const terms = getQueryTerms(query || "");
  if (!terms.length) {
    return [{ text, highlight: false }];
  }

  const regex = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "giu");
  const parts = [];
  let lastIndex = 0;
  let match = regex.exec(text);

  while (match) {
    const matchIndex = match.index;
    const matchText = match[0];

    if (matchIndex > lastIndex) {
      parts.push({
        text: text.slice(lastIndex, matchIndex),
        highlight: false
      });
    }

    parts.push({
      text: matchText,
      highlight: true
    });

    lastIndex = matchIndex + matchText.length;
    match = regex.exec(text);
  }

  if (lastIndex < text.length) {
    parts.push({
      text: text.slice(lastIndex),
      highlight: false
    });
  }

  return parts;
}

import re

# Bump whenever strip_latex's behavior changes (regex edits, new rules). Folded into the
# embedding cache key so a logic change invalidates cached embeddings even though the
# corpus's external_ids/text haven't changed - otherwise the cache would silently keep
# serving embeddings cleaned under the old, now-wrong logic.
CLEANING_VERSION = 1

# A LaTeX command with an optional single brace group (\emph{bold} -> drop both), a bare
# command (\alpha), or a math span ($...$). Not a full LaTeX parser (see quality.py's
# detection pattern) - this one removes instead of just flagging, for embedding input.
_LATEX_SPAN = re.compile(r"\\[a-zA-Z]+(\{[^{}]*\})?|\\[^a-zA-Z\s]|\$[^$]+\$")
_EXTRA_WHITESPACE = re.compile(r"\s+")


def strip_latex(text: str) -> str:
    """Remove LaTeX markup from abstract text before embedding - transformer tokenizers
    split commands like \\emph or $x^2$ into noisy subword fragments that carry no semantic
    signal for topic clustering (reports/data_quality.md: 33% of abstracts affected).
    """
    without_latex = _LATEX_SPAN.sub(" ", text)
    return _EXTRA_WHITESPACE.sub(" ", without_latex).strip()

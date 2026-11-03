from imaging_ai_atlas.data.clean_text import strip_latex


def test_strip_latex_removes_math_delimiters() -> None:
    assert strip_latex("accuracy of $x^2$ on the test set") == "accuracy of on the test set"


def test_strip_latex_removes_commands_with_braces() -> None:
    assert strip_latex(r"we use \emph{bold} markup here") == "we use markup here"


def test_strip_latex_removes_bare_commands() -> None:
    assert strip_latex(r"shows a \alpha decay curve") == "shows a decay curve"


def test_strip_latex_removes_escaped_symbols() -> None:
    assert strip_latex(r"MRI \& CT scans") == "MRI CT scans"


def test_strip_latex_leaves_plain_text_untouched() -> None:
    text = "A perfectly normal abstract with no markup at all."
    assert strip_latex(text) == text

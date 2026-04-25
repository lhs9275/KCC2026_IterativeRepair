import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from evaluation.eval_iterative import adjust_indent


def test_zero_indent_method_padded_to_four():
    code = "public void foo() {\n    return x;\n}"
    out = adjust_indent(code, 4)
    assert out == "    public void foo() {\n        return x;\n    }"


def test_already_correct_indent_unchanged():
    code = "    public void foo() {\n        return x;\n    }"
    assert adjust_indent(code, 4) == code


def test_flat_body_same_as_signature_gets_properly_nested():
    # Pathological LLM output: signature at 0, body AND closing brace at 4.
    # Previously the body_absolute_ok optimization incorrectly left body at 4
    # same level as the adjusted signature — broken Java.
    code = "public void foo() {\n    return x;\n    }"
    out = adjust_indent(code, 4)
    first, second, third = out.split("\n")
    assert first.startswith("    public")
    # Body must be DEEPER than signature, not at the same level.
    body_indent = len(second) - len(second.lstrip(" "))
    sig_indent = len(first) - len(first.lstrip(" "))
    assert body_indent > sig_indent


def test_body_strictly_deeper_uses_fast_path():
    # When body is genuinely at the target file-absolute indent (e.g., LLM
    # partially mirrored the surrounding class), we preserve that shape and
    # only fix the off-by-a-little signature.
    code = "  public void foo() {\n        return x;\n    }"
    out = adjust_indent(code, 4)
    lines = out.split("\n")
    assert lines[0].startswith("    public")  # signature re-anchored to 4
    assert lines[1].startswith("        return")  # body kept at 8
    assert lines[2].startswith("    }")  # closing brace kept at 4


def test_multi_line_body_all_shifted():
    code = "public void foo() {\n    if (x) {\n        return y;\n    }\n    return z;\n}"
    out = adjust_indent(code, 4)
    lines = out.split("\n")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines]
    # signature=4, if=8, return y=12, close-if=8, return z=8, close-method=4
    assert indents == [4, 8, 12, 8, 8, 4]


def test_negative_delta_strips_leading_spaces():
    code = "    public void foo() {\n        return x;\n    }"
    out = adjust_indent(code, 0)
    lines = out.split("\n")
    assert lines[0] == "public void foo() {"
    assert lines[1] == "    return x;"
    assert lines[2] == "}"


def test_none_code_returns_none():
    assert adjust_indent(None, 4) is None


def test_empty_code_returns_empty():
    # No non-empty lines means we can't detect a first-line indent; leave as-is.
    assert adjust_indent("", 4) == ""
    assert adjust_indent("\n\n", 4) == "\n\n"

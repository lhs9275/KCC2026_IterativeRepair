import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from evaluation.eval_iterative import _find_exact_block_span, _resolve_function_span


def test_find_exact_block_span_matches_full_method():
    file_lines = [
        "class Demo {\n",
        "    public int answer() {\n",
        "        return 42;\n",
        "    }\n",
        "}\n",
    ]
    block = "\n".join(
        [
            "    public int answer() {",
            "        return 42;",
            "    }",
        ]
    )

    assert _find_exact_block_span(file_lines, block, preferred_start=2) == (2, 4)


def test_resolve_function_span_falls_back_to_signature_and_braces():
    file_lines = [
        "class Demo {\n",
        "    public int answer() {\n",
        "        if (ready) {\n",
        "            return 42;\n",
        "        }\n",
        "        return 0;\n",
        "    }\n",
        "}\n",
    ]
    bug_meta_data = {
        "project_name": "closure-compiler",
        "function": {
            "function_before": "\n".join(
                [
                    "    public int answer() {",
                    "        // old buggy placeholder",
                    "        return -1;",
                    "    }",
                ]
            ),
            "function_before_start_line": 10,
            "function_before_end_line": 13,
            "function_after_start_line": 10,
            "function_after_end_line": 13,
        },
    }

    assert _resolve_function_span(file_lines, bug_meta_data) == (2, 7)


def test_resolve_function_span_handles_multiline_constructor_signature():
    file_lines = [
        "class Demo {\n",
        "    public Demo(int value,\n",
        "            String label) {\n",
        "        this.value = value;\n",
        "        this.label = label;\n",
        "    }\n",
        "}\n",
    ]
    bug_meta_data = {
        "project_name": "jfreechart",
        "defects4j_id": 999,
        "function": {
            "function_before": "\n".join(
                [
                    "    public Demo(int value,",
                    "            String label) {",
                    "        this.value = value;",
                    "        this.label = label;",
                    "    }",
                ]
            ),
            "function_before_start_line": 50,
            "function_before_end_line": 54,
            "function_after_start_line": 50,
            "function_after_end_line": 54,
        },
    }

    assert _resolve_function_span(file_lines, bug_meta_data) == (2, 6)

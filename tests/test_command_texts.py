"""What `ccdrift` commands, `ccdrift status`, `report` and `cost` say is worded in texts.py, not where it is printed."""

import ast
import string
from pathlib import Path

import pytest

from ccdrift import texts

SRC = Path(__file__).parents[1] / "src" / "ccdrift"


def sentences(node: ast.AST) -> list[str]:
    """The literal text in `node` that reads as words: a letter followed by a space somewhere."""
    found = []
    for part in ast.walk(node):
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            text = part.value
            if any(a.isalpha() and b == " " for a, b in zip(text, text[1:])):
                found.append(text)
    return found


@pytest.mark.parametrize("module", ["cli.py", "status.py"])
def test_no_sentence_is_printed_raised_or_returned_from_where_it_is_written(module):
    # Option help stays beside its option in cli.py; everything a command prints, the
    # errors argparse raises for it and the lines status returns come from texts.py.
    tree = ast.parse((SRC / module).read_text())
    written = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in ("print", "error"):
                written += sentences(node)
        elif isinstance(node, ast.Return) and node.value is not None:
            written += sentences(node.value)
    assert written == []


def linked_tree(module: str) -> ast.AST:
    """`module` parsed, with each `TABLE["name"]` that feeds a `.format(...)` call marked with it."""
    tree = ast.parse((SRC / module).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"
                and isinstance(node.func.value, ast.Subscript)):
            node.func.value.format_call = node
    return tree


@pytest.mark.parametrize("module", ["cli.py", "status.py", "report.py", "spend.py"])
def test_every_line_a_command_uses_exists_and_gets_exactly_its_slots(module):
    # A line is looked up by name and filled by keyword, so a misspelt name or slot would
    # fail only when that line is printed, which some error paths rarely are.
    tables = {name: getattr(texts, name) for name in ("STATUS_LINES", "COMMAND_LINES", "REPORT_LINES", "COST_LINES")}
    used = 0
    for node in ast.walk(linked_tree(module)):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in tables):
            continue
        keys = ([node.slice.value] if isinstance(node.slice, ast.Constant) else
                [branch.value for branch in (node.slice.body, node.slice.orelse)])  # "a" if x else "b"
        for key in keys:
            slots = {field for _, field, _, _ in string.Formatter().parse(tables[node.value.id][key]) if field}
            call = getattr(node, "format_call", None)
            assert slots == ({kw.arg for kw in call.keywords} if call else set()), (module, key)
            used += 1
    assert used


@pytest.mark.parametrize("module", ["status.py", "report.py", "spend.py", "settings.py", "sessions.py", "hooks.py",
                                    "failures.py"])
def test_modules_that_print_through_texts_write_no_words_of_their_own(module):
    # report and cost build their output into lists before printing it, so what they say
    # can't be told from the calls that print it: here no string but a docstring has words.
    tree = ast.parse((SRC / module).read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    words = [text for node in ast.walk(tree) if id(node) not in docstrings for text in sentences(node)
             if isinstance(node, ast.Constant)]
    assert words == []

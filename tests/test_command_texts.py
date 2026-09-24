"""What ccdrift prints, the schedule it installs and the issue drafts it writes are worded in texts.py, not
where they are printed."""

import ast
import re
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


@pytest.mark.parametrize("module", ["cli.py", "status.py", "history.py"])
def test_no_sentence_is_printed_raised_or_returned_from_where_it_is_written(module):
    # Option help stays beside its option in cli.py, and history.py builds SQL where it runs
    # it, so these are held to less than the modules below: everything they print, raise or
    # return comes from texts.py.
    tree = ast.parse((SRC / module).read_text())
    written = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in ("print", "error"):
                written += sentences(node)
        elif isinstance(node, ast.Return) and node.value is not None:
            written += sentences(node.value)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            written += sentences(node.exc)
    assert written == []


def linked_tree(module: str) -> ast.AST:
    """`module` parsed, with each `TABLE["name"]` that feeds a `.format(...)` call marked with it."""
    tree = ast.parse((SRC / module).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"
                and isinstance(node.func.value, ast.Subscript)):
            node.func.value.format_call = node
    return tree


@pytest.mark.parametrize("module", ["cli.py", "status.py", "report.py", "spend.py", "schedule.py", "draft.py",
                                    "incidents.py", "replay.py", "page.py", "state.py", "notify.py", "check.py",
                                    "logs.py", "history.py"])
def test_every_line_a_command_uses_exists_and_gets_exactly_its_slots(module):
    # A line is looked up by name and filled by keyword, so a misspelt name or slot would
    # fail only when that line is printed, which some error paths rarely are.
    tables = {name: getattr(texts, name)
              for name in ("STATUS_LINES", "COMMAND_LINES", "REPORT_LINES", "COST_LINES", "SCHEDULE_LINES",
                           "DRAFT_LINES", "INCIDENT_LINES", "REPLAY_LINES", "PAGE_LINES", "STATE_LINES",
                           "NOTIFY_LINES", "CHECK_LINES", "LOG_LINES", "HISTORY_LINES")}
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


# Every module but texts.py itself, and the two the test above holds to less; a module
# added later is held to this without anyone remembering to list it.
STRICT = sorted(path.name for path in SRC.glob("*.py") if path.name not in ("texts.py", "cli.py", "history.py"))
FORMATS = ("schedule.py", "page.py", "notify.py", "logs.py", "changelog.py")


@pytest.mark.parametrize("module", STRICT)
def test_modules_that_print_through_texts_write_no_words_of_their_own(module):
    # report and cost build their output into lists before printing it, so what they say
    # can't be told from the calls that print it: here no string but a docstring has words.
    # Module constants in FORMATS are the exception: the crontab marker, the unit files and
    # what launchctl and crontab print, the page's CSS, the AppleScript a notification runs,
    # the banners Claude Code writes and the words release notes are searched for are text
    # other programs read or write, not ccdrift's words.
    # So is markup in page.py: a piece of an f-string that is only tags once they are cut out.
    tree = ast.parse((SRC / module).read_text())
    allowed = {id(node.body[0].value) for node in ast.walk(tree)
               if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
               and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    if module in FORMATS:
        allowed |= {id(part) for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and all(isinstance(target, ast.Name) and target.id.isupper()
                            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
                    for part in ast.walk(node.value)}
    words = [text for node in ast.walk(tree) if id(node) not in allowed for text in sentences(node)
             if isinstance(node, ast.Constant)]
    if module == "page.py":
        words = [text for text in words if sentences(ast.Constant(re.sub(r"<[^<>]*(>|$)", "", text)))]
    assert words == []

"""Keep the line numbers in doc/ <-> code cross-references accurate.

Explanation of the scheme, and why the prose lives in doc/ at all:
See doc/doc-refs.md

    python scripts/doc_refs.py            # check, exit 1 if anything is stale
    python scripts/doc_refs.py --fix      # rewrite the numbers, then check
"""

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "doc"

# In a doc: "Code: `scripts/train_vqgan.py:259 :: build_opt_g()`" — the path and
# symbol are written by hand, the line number is generated.
DOC_TO_CODE = re.compile(
    r"`(?P<path>[\w./-]+\.py)(?::(?P<line>\d+))? :: (?P<symbol>[\w.]+)(?P<call>\(\))?`"
)

# In code: "# See doc/clocks.md:46 (budgets)" — the file and slug are written by
# hand, the line number is generated. Without a slug the whole doc is the answer
# and there is no number to keep.
CODE_TO_DOC = re.compile(
    r"doc/(?P<name>[\w-]+\.md)(?::(?P<line>\d+))?(?:\s+\((?P<slug>[\w-]+)\))?"
)


def py_files():
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def symbol_lines(path: Path):
    """{"build_opt_g": 259, "TrainingDisplay.set_losses": 128, ...}"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    out = {}

    def walk(node, prefix=""):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                out.setdefault(name, child.lineno)
                out.setdefault(child.name, child.lineno)
                walk(child, name + ".")

    walk(tree)
    return out


def heading_lines(path: Path):
    """{"budgets": 42, ...} — GitHub-style slugs of every markdown heading."""
    out = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("#"):
            continue
        text = line.lstrip("#").strip()
        slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
        slug = re.sub(r"\s+", "-", slug)
        if slug:
            out.setdefault(slug, i)
    return out


def check(fix=False):
    problems, fixed = [], 0
    sym_cache, head_cache = {}, {}

    def symbols(rel):
        if rel not in sym_cache:
            sym_cache[rel] = symbol_lines(ROOT / rel)
        return sym_cache[rel]

    def headings(name):
        if name not in head_cache:
            head_cache[name] = heading_lines(DOC / name)
        return head_cache[name]

    # ---- doc -> code ----
    for doc in sorted(DOC.glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        out, last = [], 0
        for m in DOC_TO_CODE.finditer(text):
            rel, sym, want = m["path"], m["symbol"], m["line"]
            if not (ROOT / rel).exists():
                problems.append(f"{doc.name}: no such file {rel}")
                continue
            got = symbols(rel).get(sym)
            if got is None:
                problems.append(f"{doc.name}: {rel} has no {sym}")
                continue
            if want != str(got):
                if not fix:
                    problems.append(
                        f"{doc.name}: {rel} :: {sym} says :{want}, is :{got}"
                    )
                else:
                    out.append(text[last:m.start()])
                    out.append(f"`{rel}:{got} :: {sym}{m['call'] or ''}`")
                    last = m.end()
                    fixed += 1
        if fix and out:
            out.append(text[last:])
            doc.write_text("".join(out), encoding="utf-8", newline="\n")

    # ---- code -> doc ----
    for py in py_files():
        text = py.read_text(encoding="utf-8")
        out, last = [], 0
        for m in CODE_TO_DOC.finditer(text):
            name, slug, want = m["name"], m["slug"], m["line"]
            if not (DOC / name).exists():
                problems.append(f"{py.relative_to(ROOT)}: no such doc {name}")
                continue
            if slug is None:
                if want is not None:
                    problems.append(
                        f"{py.relative_to(ROOT)}: doc/{name}:{want} has no (slug) to"
                        f" anchor the number to"
                    )
                continue
            got = headings(name).get(slug)
            if got is None:
                problems.append(f"{py.relative_to(ROOT)}: {name} has no heading ({slug})")
                continue
            if want != str(got):
                if not fix:
                    problems.append(
                        f"{py.relative_to(ROOT)}: doc/{name} ({slug}) says :{want},"
                        f" is :{got}"
                    )
                else:
                    out.append(text[last:m.start()])
                    out.append(f"doc/{name}:{got} ({slug})")
                    last = m.end()
                    fixed += 1
        if fix and out:
            out.append(text[last:])
            py.write_text("".join(out), encoding="utf-8", newline="\n")

    return problems, fixed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true", help="rewrite stale line numbers")
    args = ap.parse_args(argv)

    if args.fix:
        _, fixed = check(fix=True)
        print(f"updated {fixed} reference(s)")

    problems, _ = check(fix=False)
    for p in problems:
        print(p)
    if problems:
        print(f"\n{len(problems)} stale or broken reference(s)")
        return 1
    print("all doc/code references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())

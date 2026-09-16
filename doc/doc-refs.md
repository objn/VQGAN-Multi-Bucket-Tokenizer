# Where the prose lives, and how the references stay true

Code: `scripts/doc_refs.py:73 :: check()`

Explanations do not live in the `.py` files. A file whose comments outweigh its
code — `vqgan/config.py` was 64% comment lines — is one where neither the code
nor the explanation can be read without skipping past the other. So the prose
is here, one file per topic, and the code keeps a one-line pointer.

What stays in the code is what a reader needs *at that line*: what the block
does, and where to go for why.

## The two reference forms

**doc → code**, in each doc's `Code:` header:

```
Code: `scripts/train_vqgan.py:175 :: build_opt_g()`
```

The path and the symbol are written by hand. The line number is generated —
`doc_refs.py` resolves the symbol through the AST.

**code → doc**, in a comment:

```python
# Each budget against its own clock. See doc/clocks.md:78 (budgets)
```

The file and the `(slug)` are written by hand; the slug is a GitHub-style
anchor for one of the doc's headings. The line number is generated from where
that heading currently sits.

A reference with no slug means the whole document is the answer, and carries
no number to keep:

```python
# Three clocks. See doc/clocks.md
```

## Keeping them accurate

```
python scripts/doc_refs.py          # check; exit 1 if anything is stale
python scripts/doc_refs.py --fix    # rewrite the numbers, then check
```

Line numbers go stale the moment anything above them moves, which is the known
cost of asking for them. Nothing here is hand-maintained: the durable half of
each reference is a *name* — a symbol or a heading slug — and the number is
derived from it. Run `--fix` before committing and the numbers are right;
run the plain check in review and a rename or a deleted heading shows up as a
broken reference rather than as a number that quietly points at the wrong
line.

The checker also fails on a reference to a file that does not exist, a symbol
that is gone, a heading slug that no longer matches, and a number written
without a slug to anchor it.

## Writing a new topic

1. Create `doc/<topic>.md` with a `Code:` header naming the symbols it
   explains.
2. Replace the prose in the code with a short line ending
   `See doc/<topic>.md (<heading-slug>)`.
3. Run `python scripts/doc_refs.py --fix`.

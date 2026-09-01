"""Build and execute Jupyter notebooks from `# %%` percent-format Python sources.

Usage:  python src/nbbuild.py src/nb01_audit.py notebooks/01_audit.ipynb
Cell markers:
    # %% [markdown]      -> markdown cell (body lines are `# ` prefixed)
    # %%                 -> code cell
"""
import sys, re, pathlib
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor


def parse(src_text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            body = "\n".join(
                re.sub(r"^# ?", "", ln) for ln in body.split("\n")
            ).strip("\n")
            cells.append(new_markdown_cell(body))
        else:
            cells.append(new_code_cell(body))

    for line in src_text.split("\n"):
        s = line.rstrip()
        if s.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in s else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def build(src, dst, execute=True, timeout=1800):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    nb = new_notebook(cells=parse(src.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.metadata["language_info"] = {"name": "python"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(dst.parent.resolve())}})
    nbformat.write(nb, dst)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"[nbbuild] {dst}  cells={len(nb.cells)} (md={n_md}, code={len(nb.cells)-n_md}) executed={execute}")


if __name__ == "__main__":
    a = sys.argv[1:]
    ex = "--no-exec" not in a
    a = [x for x in a if not x.startswith("--")]
    build(a[0], a[1], execute=ex)

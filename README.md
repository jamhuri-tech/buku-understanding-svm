# Code: Understanding Support Vector Machines

Companion repository for ***Understanding Support Vector Machines: The
Widest Margin, One Vector at a Time*** (First Edition, 2026) by Mohammad
Jamhuri. The tag `edition-1` marks the code used for that edition.

The book follows one small data set from start to finish: eight machines
in `code/machines.py`, each with two measurements and a label. Their
hard-margin solution is w = (1, 2), b = -11, with support vectors at
machines 4, 5, and 7 and multipliers 5/2, 2, and 1/2. Chapter 19 closes
the book on the Wine data from the UCI Machine Learning Repository
(CC BY 4.0, DOI 10.24432/C5PC7J), loaded through scikit-learn.

**Every output block printed in the book comes from the code here.**
`check_outputs.py` reruns each chapter and compares the result with the
blocks printed in the book, which are stored in `outputs/chNN.txt`.

All code is free to use, copy, change, and share for any purpose under
the [0BSD](LICENSE) licence, with no duty to credit the source.

## Running

```bash
git clone https://github.com/jamhuri-tech/buku-understanding-svm.git
cd buku-understanding-svm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd code
../.venv/bin/python ch01_walkthrough.py
```

Run the chapter scripts from inside `code/`, because they import shared
modules from that folder.

## Layout

| Path | Contents |
| --- | --- |
| `code/chNN_*.py` | code of Chapter NN |
| `code/machines.py` | the eight machines |
| `code/search_mini.py` | the search that picked the eight machines |
| `code/ch12_timing.json`, `code/ch13_timing.json` | timing results used by the figures |
| `gen_fig.py` | rebuilds the Matplotlib figures into `fig/` |
| `gen_cover.py` | rebuilds the cover plot |
| `outputs/chNN.txt` | output blocks as printed in the book |
| `check_outputs.py` | compares the code with those blocks |

## Checking the book

```bash
.venv/bin/python check_outputs.py
.venv/bin/python check_outputs.py 04 05
```

The printed blocks were produced on macOS with the versions in
`requirements.txt`, and the automatic check runs on the same kind of
machine. On another CPU, rounding can steer an iterative solver down a
slightly different path, so step counts, support-vector counts near a
tolerance, and the last digits of some numbers may differ, while the
solutions agree within the solver tolerance.

A full run takes several minutes. The first form covers all chapters,
and the second covers the chapters you name.

Timing results depend on the computer, so they are stored as JSON and
not rerun by the check. To measure them again, run
`python ch12_smo.py timing` or `python ch13_linear.py timing` inside `code/`.

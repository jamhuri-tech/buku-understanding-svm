"""Check that the code in this repository produces the numbers in the book.

For each outputs/chNN.txt, every code/chNN_*.py is run and the outputs
are joined. Each block printed in the book must then appear whole inside
that text. Repeated spaces, the sign of a printed zero, digits past the
tenth decimal, and numbers below 1e-9 are ignored, since these are
rounding noise that depends on the CPU.

    python check_outputs.py          # all chapters
    python check_outputs.py 04 05    # selected chapters
"""
import difflib
import glob
import os
import re
import subprocess
import sys


def tidy(text):
    """Write rounding noise in one form."""
    text = re.sub(r"\b\d(?:\.\d+)?e-(?:09|[1-9]\d)\b", "<1e-9", text)
    text = re.sub(r"-0\.(?=[\s\]])", "0.", text)
    text = re.sub(r" +", " ", text)  # NumPy pads to the widest entry
    text = re.sub(r"(\d\.\d{10})\d{3,}", r"\1", text)
    # The diverging solver of Chapter 6 stops at a size set by rounding.
    return re.sub(r"sum alpha \d\.\de\+1\d", "sum alpha <huge>", text)


chosen = sys.argv[1:]
problems = 0
for path in sorted(glob.glob("outputs/ch[0-9][0-9].txt")):
    nn = path[-6:-4]
    if chosen and nn not in chosen:
        continue
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"^=== block \d+ ===\n", text, flags=re.M)[1:]
    output = ""
    for script in sorted(glob.glob(f"code/ch{nn}_*.py")):
        run = subprocess.run([sys.executable, os.path.basename(script)],
                             cwd="code", capture_output=True, text=True)
        if run.returncode != 0:
            print(f"{script}: FAILED\n{run.stderr[-500:]}")
            problems += 1
        output += run.stdout
    clean = tidy("\n".join(line.rstrip() for line in output.splitlines()))
    matched = 0
    for block in blocks:
        wanted = tidy("\n".join(line.rstrip()
                                for line in block.strip("\n").splitlines()))
        if wanted in clean:
            matched += 1
        else:
            problems += 1
            print(f"Chapter {int(nn)}: block not produced by the code:")
            have = clean.splitlines()
            for line in wanted.splitlines():
                if line not in have:
                    near = difflib.get_close_matches(line, have, 1, 0.5)
                    print(f"    book: {line}")
                    print(f"    code: {near[0] if near else '(no close line)'}")
    print(f"Chapter {int(nn)}: {matched}/{len(blocks)} blocks match")

print("ALL MATCH" if problems == 0 else f"{problems} problems")
sys.exit(1 if problems else 0)

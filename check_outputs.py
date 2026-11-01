"""Check that the code in this repository produces the numbers in the book.

For each outputs/chNN.txt, every code/chNN_*.py is run and the outputs
are joined. Each block printed in the book must then appear whole inside
that text. Trailing spaces are ignored, and numbers such as 2.8e-17
(rounding noise that depends on the CPU) count as equal below 1e-9.

    python check_outputs.py          # all chapters
    python check_outputs.py 04 05    # selected chapters
"""
import glob
import os
import re
import subprocess
import sys


def tidy(text):
    """Write rounding noise below 1e-9 in one form."""
    return re.sub(r"\b\d(?:\.\d+)?e-(?:09|[1-9]\d)\b", "<1e-9", text)


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
            print("    " + wanted.replace("\n", "\n    ")[:400])
    print(f"Chapter {int(nn)}: {matched}/{len(blocks)} blocks match")

print("ALL MATCH" if problems == 0 else f"{problems} problems")
sys.exit(1 if problems else 0)

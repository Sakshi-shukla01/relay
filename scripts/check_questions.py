"""Check every expected answer phrase in data/questions.json literally occurs
in the fetched paper text. The questions were drafted with an AI assistant
from facts it extracted from the PDFs; this is how we catch it being wrong
before we grade anything against it.

  python scripts/check_questions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import DATA, RESULTS  # noqa: E402
from relay.corpus import load_pages  # noqa: E402
from relay.grading import has, norm  # noqa: E402

qs = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
pages = load_pages()
text = {k: norm(" ".join(v)) for k, v in pages.items()}
lines, bad = ["| q | paper | variant found in paper text? |", "|---|---|---|"], 0
for q in qs:
    if not q["answerable"]:
        continue
    found = [v for v in q["expect"] if all(has(text[q["paper"]], p) for p in v)]
    ok = bool(found)
    bad += not ok
    lines.append(f"| {q['id']} | {q['paper']} | {'yes: ' + ' + '.join(found[0]) if ok else 'NO - fix this question'} |")
out = "\n".join(lines) + f"\n\n{bad} answerable question(s) have no expected phrase in the paper text.\n"
(RESULTS / "question_check.md").write_text(out, encoding="utf-8")
print(out)
sys.exit(1 if bad else 0)

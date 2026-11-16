# Checklist (delete this file before you submit)

Deadline: **Fri 13 Nov 2026, 5:11 PM IST**.

## Day 1 (today): run it
1. Make a free Groq account and API key at console.groq.com. Keep the key out of every file.
2. Set up:
   ```bash
   cd relay
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   export GROQ_API_KEY=gsk_...        # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
   ```
3. `python run_all.py --stages papers`. Check that it prints 3 papers and that
   `results/question_check.md` says **0** questions are missing. If one is missing,
   open the paper, fix or drop the question in `data/questions.json`, and note it
   (it's your "the assistant got something wrong" story).
4. `python run_all.py --stages probe,calibrate,offline,sweep,standin,analyze` (~60 min).
5. Look at `results/tables.md`. Does anything surprise you? Look at a few answers
   in `results/traces.jsonl`. If the grader is wrong about one, add it to
   `data/grade_overrides.json` as `{"eval_main:A07": "correct"}` and say so in the write-up.

## Day 2: the 120b runs (they get a fresh daily budget)
6. `python run_all.py --stages sweep_large,eval,real_load,analyze`

## Write-up
7. Fill every ⟨FILL⟩ in `WRITEUP_DRAFT.md` from `results/tables.md`. Paste the tables in.
8. Write the YOURS sections yourself: Vaani feedback (visit the site), the thing you learned,
   and "How I worked with the assistant". Export this conversation as it is, unedited.
9. Keep it to 2 pages of writing (the tables don't count toward that).
10. Delete the `<!-- -->`/⟨⟩ notes, `WRITEUP_DRAFT` → `WRITEUP`, and this checklist.

## Google Drive folder
- `WRITEUP.pdf` (or a Google Doc)
- `results.csv` at the top level (copy it out of `results/`), plus the `results/` and `raw/` folders
- the code (`relay/`, `scripts/`, `run_all.py`, `README.md`, `requirements.txt`, `data/questions.json`)
- the conversation export
- Share settings: "Anyone with the link can view"
- Before uploading, search everything for `gsk_`. It must find nothing.

## Be ready for the live call
They will ask you to change something live. Practise these:
- Halve the 120b TPM in `relay/config.py` (DOCUMENTED_LIMITS) and re-run `rpm_halved`.
- Change the patience range in `scripts/run_load.py` and explain how the given-up % moves.
- Swap the retrieval default in `relay/config.py` (`DEFAULT_RETRIEVAL`).
- Explain why L1 drops the verifier before L2 drops the 120b drafter.

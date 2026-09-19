# AI Application Development — Practice Tests

Five 10-question multiple-choice practice tests for CIS-305 (AI Application Development,
Mercyhurst, Fall 2026). Static site, no build step for the pages — GitHub Pages serves it as is.

- `index.html` — home page: the five tests + links to the solutions
- `test.html?t=N` — take test N. Submitting shows the score and marks missed questions
  **without revealing the correct answer**. Students can retry just the missed ones.
- `solutions.html?t=N` — worked solutions, including why each wrong choice is wrong
- `questions.js` — **generated**; do not edit by hand
- `tools/build_questions.py` — the source of truth for all 50 questions

## What's on the tests

Two questions per test from each of five topics:

| Topic | Source |
|---|---|
| Chunking by hand | textbook practice set |
| Chunking edge cases | textbook practice set |
| Precision & recall in search | textbook practice set |
| Building the flow: load & store | Building RAG, part 1 |
| Building the flow: the RAG app | Building RAG, part 2 |

## Editing questions

Edit `tools/build_questions.py`, then rebuild:

    ~/.langflow/.langflow-venv/bin/python tools/build_questions.py   # also checks chunking against the real Langflow splitter
    python3 tools/build_questions.py                                 # works anywhere; built-in splitter check only

The build asserts every chunk length and every precision/recall answer before writing
`questions.js`, so a typo in an answer key fails loudly.

## Notes

- The answer key is in `questions.js`, so a determined student can find it in the page source.
  That's acceptable here — the solutions are published anyway. The point is that the results
  screen never shows them.
- Progress and "best score" live in the student's browser (localStorage). Nothing is sent anywhere.
- Everything is answerable with pencil and paper — no devices, same as the real exams.

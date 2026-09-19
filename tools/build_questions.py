"""
build_questions.py — the ONE source of truth for all 50 practice questions.

Run it after any edit:

    python3 tools/build_questions.py

It checks every answer, then writes ../questions.js (the file the site loads).
Never edit questions.js by hand; it gets overwritten.

What gets checked:
  * every chunking question is re-run through the splitting algorithm, and —
    if you run it with the Langflow Desktop Python, which has langchain
    installed — through the REAL CharacterTextSplitter as well:

        ~/.langflow/.langflow-venv/bin/python tools/build_questions.py

  * precision and recall answers are recomputed here and asserted;
  * each question has exactly 4 choices, one correct, no duplicates;
  * answer letters are spread evenly across A–D.

Question format
  choices: list of (text, why). The FIRST entry is the correct answer and its
           `why` is None. Every other entry's `why` explains the specific
           mistake that produces it (shown on the solutions page only).
  pos:     the letter the correct answer should land on. The build moves the
           correct choice there and keeps the distractors in listed order.
  doc:     (chunking only) (text, separator, chunk_size, chunk_overlap,
           expected_chunk_lengths). The page shows the document with a
           character count beside every line, like the textbook does.

Everything is answerable with pencil and paper — no calculator, no devices,
same as the real exam.
"""

import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "questions.js")

N, NN = "\n", "\n\n"


# ------------------------------------------------------------------ helpers
def fr(a, b):
    """Stacked fraction for display."""
    return f'<span class="fr"><span>{a}</span><span>{b}</span></span>'


def pct(num, den):
    return round(100 * num / den)


def merge(text, sep, size, overlap):
    """langchain CharacterTextSplitter._merge_splits, keep_separator=False."""
    atoms = [a for a in text.split(sep) if a != ""]
    sl, cur, total, chunks = len(sep), [], 0, []
    for a in atoms:
        n = len(a)
        if total + n + (sl if cur else 0) > size and cur:
            chunks.append(sep.join(cur).strip())
            while total > overlap or (total + n + (sl if cur else 0) > size and total > 0):
                total -= len(cur[0]) + (sl if len(cur) > 1 else 0)
                cur = cur[1:]
        cur.append(a)
        total += n + (sl if len(cur) > 1 else 0)
    chunks.append(sep.join(cur).strip())
    return [c for c in chunks if c]


try:
    import logging
    logging.disable(logging.CRITICAL)
    from langchain_text_splitters import CharacterTextSplitter
    HAVE_LANGCHAIN = True
except Exception:
    HAVE_LANGCHAIN = False


def check_chunks(text, sep, size, overlap, expected):
    got = [len(c) for c in merge(text, sep, size, overlap)]
    assert got == expected, f"chunk lengths {got} != expected {expected}"
    if HAVE_LANGCHAIN:
        real = CharacterTextSplitter(separator=sep, chunk_size=size, chunk_overlap=overlap,
                                     keep_separator=False).split_text(text)
        assert [len(c) for c in real] == expected, f"REAL splitter gave {[len(c) for c in real]}"


def shared_lines(text, sep, size, overlap):
    ch = merge(text, sep, size, overlap)
    return [len(set(ch[i].split(sep)) & set(ch[i - 1].split(sep))) for i in range(1, len(ch))]


# ------------------------------------------------------------------ topics
CHK = "Chunking by hand"
EDGE = "Chunking edge cases"
PR = "Precision & recall in search"
LOAD = "Building the flow: load & store"
RAG = "Building the flow: the RAG app"

Q = []  # every question, in test order


def q(test, topic, pos, prompt, choices, solution, doc=None, settings=None):
    Q.append(dict(test=test, topic=topic, pos=pos, prompt=prompt, choices=choices,
                  solution=solution, doc=doc, settings=settings))


# ================================================================== LOAD & STORE
# Section 23: Read File → Split Text → Chroma DB (Ingest Data), Embedding Model
# → Chroma DB (Embedding), Chat Input → Search Query, Search Results → Chat Output.

q(1, LOAD, "B",
  "<p>You have wired <strong>Read File → Split Text</strong>. Which connection puts the document into the vector store?</p>",
  [("Split Text's <em>Chunks</em> output into Chroma DB's <em>Ingest Data</em> input", None),
   ("Split Text's <em>Chunks</em> output into Chroma DB's <em>Search Query</em> input",
    "<em>Search Query</em> is the reading side — it takes the question you type in the Playground. Sending chunks there searches the store instead of filling it."),
   ("Read File's <em>Loaded Files</em> output into Chroma DB's <em>Ingest Data</em> input",
    "That skips the chunker. The whole document would go in as one piece, and every search would return all of it."),
   ("The Embedding Model's <em>Embeddings</em> output into Chroma DB's <em>Ingest Data</em> input",
    "The Embedding Model supplies the model that turns text into points; it connects to Chroma's <em>Embedding</em> input, and it carries no document text.")],
  "<p>Loading runs Read File → Split Text → Chroma DB. The chunks are the data being stored, so they go into <strong>Ingest Data</strong>.</p>"
  "<p>The Embedding Model connects separately, into <em>Embedding</em>, because Chroma needs a way to turn both the chunks and later the question into points.</p>")

q(1, LOAD, "D",
  "<p>You need an embedding model for the store, and you look for Anthropic. It is not an option. What is going on?</p>",
  [("Anthropic makes chat models but no embedding model — use the OpenAI embedder instead", None),
   ("Your Anthropic key was never saved, so the option is hidden",
    "A missing key gives an error when you run the component; it does not hide the provider. Anthropic offers no embedding model at all."),
   ("Anthropic's embedder only works with Anthropic's chat models, so it is hidden until you add one",
    "Embedding and chat are separate jobs, and nothing pairs them. You can embed with one company's model and write answers with another's."),
   ("Embedding happens inside Chroma, so no embedding model is needed at all",
    "Chroma stores points and compares them, but something has to turn the text into points first. That is the embedder's job.")],
  "<p>Embedding and chat are different jobs, and Anthropic sells chat models only.</p>"
  "<p>Use the <strong>OpenAI embedder</strong> with your OpenAI key. The chunks and the questions then land in the same space, which is what makes the search work.</p>")

q(2, LOAD, "A",
  "<p>Your flow is built and the wires look right, but the Playground returns nothing at all — an empty result, every time. What is the most likely cause?</p>",
  [("You never clicked <em>Run component</em> on Chroma DB, so the store is empty", None),
   ("Your question uses words that do not appear in the document",
    "Search matches on meaning, not shared words — that is the whole point of embeddings. A wording mismatch changes <em>which</em> chunk comes back, not whether anything comes back."),
   ("Number of Results is set too low",
    "Even k = 1 returns one chunk. A low k narrows the results; it cannot empty them."),
   ("Persist Directory is empty",
    "An empty Persist Directory only means the store lives in memory for this session. It still stores and still searches.")],
  "<p>The flow does two jobs. <em>Run component</em> on Chroma DB <strong>loads</strong> the chunks; the Playground <strong>searches</strong> what is already loaded.</p>"
  "<p>An empty store returns empty results. Load first, then search.</p>")

q(2, LOAD, "C",
  "<p>You leave Chroma DB's <strong>Persist Directory</strong> empty. What does that mean for your stored chunks?</p>",
  [("They live in memory for this session, so you have to load them again next time", None),
   ("Langflow picks a default folder, so they are saved permanently",
    "Langflow does not fill the field in for you. Blank means nothing is written to disk."),
   ("The flow refuses to run until you choose a folder",
    "The field is optional. The flow runs fine — the store just is not saved."),
   ("The chunks are saved but their embeddings are not",
    "The store keeps the text and the points together. Either both are saved or neither is.")],
  "<p>Leave Persist Directory empty and the store is in memory only: convenient for class, gone when the session ends.</p>"
  "<p>Fill it in and the same chunks are still there next time, with no reloading.</p>")

q(3, LOAD, "B",
  "<p>You fix a typo in your document and save the file. What has to happen before a search can find the corrected text?</p>",
  [("Click <em>Run component</em> on Chroma DB again to reload the document", None),
   ("Nothing — the Playground reads the file again with every question",
    "The Playground only searches what is already in the store. Reading and chunking the file happen on the loading side of the flow."),
   ("Run the Split Text component, since that is where the document gets read",
    "Running Split Text produces fresh chunks but leaves them sitting there. Only running Chroma DB puts them into the store."),
   ("Restart Langflow so it picks up the new file",
    "Restarting clears an in-memory store and still does not load anything. You would have to run the component anyway.")],
  "<p>Loading is a step you trigger, not something that happens continuously. Edit the document, then <strong>Run component</strong> on Chroma DB to load it again.</p>")

q(3, LOAD, "D",
  "<p>Which part of the flow runs <strong>once</strong>, and which part runs <strong>every time</strong> you ask a question?</p>",
  [("Once: Read File → Split Text → Chroma DB. Every question: Chat Input → Chroma DB → Chat Output", None),
   ("Once: Chat Input → Chroma DB. Every question: Read File → Split Text → Chroma DB",
    "Backwards. The document is loaded once; the question is asked over and over."),
   ("Both parts run once, when you click Run component",
    "Then you could only ever ask one question. Searching happens again for every question you type."),
   ("Both parts run every question, which is why loading a big document is slow",
    "If the document were re-read and re-chunked for every question you would pay to embed it again each time. That is exactly what loading once avoids.")],
  "<p>Two jobs, one flow. <strong>Loading</strong> — read, split, embed, store — happens once. <strong>Searching</strong> — question in, chunks out — happens for every question.</p>")

q(4, LOAD, "C",
  "<p>What is the Embedding Model component's job in the flow?</p>",
  [("It supplies the model that turns text into points — used on the chunks when loading and on the question when searching", None),
   ("It turns only the question into a point; the chunks are stored as plain text",
    "Then there would be nothing to compare the question against. The chunks are turned into points at load time — that is what makes them searchable."),
   ("It turns only the chunks into points; the question is matched by keyword",
    "Both sides have to be points, or there is no way to measure the distance between them."),
   ("It writes the answer from the chunks that come back",
    "That is the Language Model's job, and it is not in the flow at all until you add generation.")],
  "<p>One Embedding Model connects to Chroma's <em>Embedding</em> input and serves both sides: every chunk becomes a point when you load, and every question becomes a point when you search.</p>"
  "<p>Retrieval is then just finding the stored points nearest to the question's point.</p>")

q(4, LOAD, "A",
  "<p>You load your chunks with one embedding model. Later you switch the Embedding Model component to a different model and search without reloading. What happens?</p>",
  [("The results are junk: the question's point is in a different space than the stored points", None),
   ("Chroma converts the stored points to the new model automatically",
    "There is no conversion. A point only means something to the model that produced it."),
   ("Langflow stops with an error saying the models do not match",
    "Nothing complains. The numbers still have the right shape, so you get confident-looking nonsense — the worst kind of bug."),
   ("It works, but every search is slower",
    "Speed is not the problem. The comparison itself is meaningless.")],
  "<p>Two different models put text in two different spaces, so distances between them mean nothing.</p>"
  "<p>Use the same embedding model on both sides. If you change it, load the document again.</p>")

q(5, LOAD, "B",
  "<p>In the search half of the flow, what arrives at Chroma DB's <strong>Search Query</strong> input?</p>",
  [("The question the student typed, coming from Chat Input", None),
   ("The chunks that Split Text produced",
    "Those go to <em>Ingest Data</em>. Search Query is the reading side of the store."),
   ("The retrieved chunks, on their way back out",
    "Retrieved chunks leave the store through the <em>Search Results</em> output; nothing comes back in."),
   ("The output of the Embedding Model",
    "The Embedding Model connects to the <em>Embedding</em> input. It supplies the model, not the text being looked up.")],
  "<p>Chat Input carries the question into <strong>Search Query</strong>. Chroma turns that question into a point and returns the nearest stored chunks through <em>Search Results</em>.</p>")

q(5, LOAD, "D",
  "<p>Which statement about the cost of embeddings is correct?</p>",
  [("You pay to embed every chunk each time you load the document, plus a tiny amount to embed each question", None),
   ("Embeddings are free; only the chat model costs anything",
    "Embedding is a model call like any other. It is cheap, not free — and reloading a big document over and over does add up."),
   ("You pay per search, based on how much text is in the store",
    "Searching compares points that already exist. The size of the store is not what you are billed for."),
   ("Embedding a document costs more than the answers the chat model writes",
    "It is the other way around: embedding models are far cheaper per token than chat models.")],
  "<p>Loading is the expensive moment, and it is a one-time cost per load: every chunk gets embedded. Each question then costs one small embedding.</p>"
  "<p>Load once and search freely; reloading a big document repeatedly is what runs up a bill.</p>")


# ================================================================== THE RAG APP
# Section 24: Search Results → Parser → Prompt Template {context} → Language
# Model (System Message); Chat Input → Language Model (Input) → Chat Output.

q(1, RAG, "C",
  "<p>Why does a <strong>Parser</strong> sit between Chroma DB and the Prompt Template?</p>",
  [("Retrieved chunks arrive as structured rows with metadata; Parser turns them into plain text for the prompt", None),
   ("It shortens the chunks so they fit inside the context window",
    "Nothing is trimmed. What controls how much text arrives is your chunk size and Number of Results."),
   ("It ranks the chunks so the best one goes first",
    "The ranking is already done — Chroma returns them nearest-first. Parser only changes the format."),
   ("It strips the question out of the retrieved text",
    "The question never went into the store. It reaches the model by a separate wire from Chat Input.")],
  "<p>Search Results are structured data — rows with a text column plus metadata. A prompt needs plain text.</p>"
  "<p>Set the Parser's mode to <em>Parser</em> with the template <code>{text}</code> and it hands over just the chunk text, one per line.</p>")

q(1, RAG, "A",
  "<p>Your prompt says: “Answer using only the context below. Context: <code>{context}</code>”. When a student asks a question, what has been put in that blank?</p>",
  [("The text of the chunks retrieved for that question", None),
   ("The whole document",
    "That is what you did when you chatted with a document by pasting it into the prompt. Retrieval exists so the model reads a page or two instead."),
   ("Every chunk in the store",
    "Only the top few come back — as many as Number of Results asks for."),
   ("The student's question",
    "The question reaches the model on its own wire from Chat Input. The blank is for the material the answer has to come from.")],
  "<p>Each question searches the store, and the chunks that come back are turned into plain text and dropped into the blank.</p>"
  "<p>So the prompt is rebuilt for every question — same instruction, different context.</p>")

q(2, RAG, "D",
  "<p>In the finished RAG app, where do the Prompt Template's output and the Chat Input connect on the Language Model?</p>",
  [("Prompt Template → <em>System Message</em>, and Chat Input → <em>Input</em>", None),
   ("Prompt Template → <em>Input</em>, and Chat Input → <em>System Message</em>",
    "Swapped. The standing instructions plus the retrieved context are the system message; the student's question is the input."),
   ("Both go to <em>Input</em>, one after the other",
    "An input takes one wire. The instruction and the question travel separately."),
   ("Prompt Template → <em>System Message</em>, and Chroma DB's Search Results → <em>Input</em>",
    "The retrieved chunks reach the model inside the prompt, through the Parser and <code>{context}</code>. The <em>Input</em> is for the question.")],
  "<p>The Prompt Template holds the instruction and the retrieved context, so it goes to <strong>System Message</strong>. The question itself comes from Chat Input into <strong>Input</strong>.</p>")

q(2, RAG, "B",
  "<p>In the finished RAG app, Chat Input feeds <strong>two</strong> components. Which two?</p>",
  [("Chroma DB's <em>Search Query</em> and the Language Model's <em>Input</em>", None),
   ("Chroma DB's <em>Search Query</em> and the Prompt Template",
    "The prompt receives the retrieved context, not the question. The question goes straight to the model."),
   ("The Parser and the Language Model",
    "The Parser only ever handles what comes back out of the store."),
   ("Chroma DB's <em>Search Query</em> and the Chat Output",
    "Chat Output shows the model's answer. Echoing the question there would do nothing useful.")],
  "<p>The question does two jobs: it is what the store gets searched with, and it is what the model is asked.</p>"
  "<p>So Chat Input fans out — one wire to <strong>Search Query</strong>, one wire to the model's <strong>Input</strong>.</p>")

q(3, RAG, "A",
  "<p>Your RAG app answers from the model's general knowledge instead of from the document. The wiring is correct and the store is loaded. What is the most likely cause?</p>",
  [("The prompt never says to use <em>only</em> the context", None),
   ("The Parser template is <code>{text}</code> instead of <code>{context}</code>",
    "Those names belong to different components. <code>{text}</code> is what the Parser pulls from each row; <code>{context}</code> is the blank in the prompt."),
   ("Number of Results is too high, so the model has too much to read",
    "More chunks can add noise, but a grounded prompt still answers from them. Extra context does not send the model to its own memory."),
   ("The Language Model is a chat model rather than an embedding model",
    "Writing the answer is exactly a chat model's job. An embedding model cannot write at all.")],
  "<p>Without the rule, the model treats the context as helpful background and falls back on what it already knows.</p>"
  "<p>Say it plainly: answer using <strong>only</strong> the context below, and if the answer is not there, say you don't know.</p>")

q(3, RAG, "C",
  "<p>After the rewire, what does Chat Output display?</p>",
  [("The answer the Language Model wrote", None),
   ("The retrieved chunks, exactly as they came out of the store",
    "That was the part 1 flow, before generation was added. The wire from Search Results to Chat Output is removed during the rewire."),
   ("The filled-in prompt, so you can see what the model was sent",
    "The prompt goes to the model, not to the screen. To see it, use Inspect output on the Prompt Template."),
   ("The chunk text from the Parser",
    "The Parser feeds the prompt. Its text reaches you only after the model has used it to write an answer.")],
  "<p>The chain ends at the model: Chroma → Parser → Prompt Template → Language Model → <strong>Chat Output</strong>.</p>"
  "<p>The chunks are still doing the work, but now the student sees a written answer instead of raw text.</p>")

q(4, RAG, "B",
  "<p>A student asks your handbook app something the handbook never covers. With the grounding-and-honesty prompt in place, what <em>should</em> happen?</p>",
  [("The model says it does not know, because the answer is not in the retrieved context", None),
   ("The model answers from general knowledge, with a note that it is guessing",
    "The rule says to use only the context. An answer with a disclaimer attached is still an answer from outside the document."),
   ("Chroma returns nothing, so the app shows an empty reply",
    "The store still returns its nearest chunks — nearest is not the same as relevant. The honesty rule is what keeps them from being used badly."),
   ("The app errors, because the prompt's <code>{context}</code> is empty",
    "The context is not empty; it holds chunks that happen not to answer the question.")],
  "<p>Retrieval always returns <em>something</em> — the closest chunks it has. Judging whether they actually answer the question is the prompt's job.</p>"
  "<p>That is the behavior worth testing: ask about the Wi-Fi password and see whether the app admits it does not know.</p>")

q(4, RAG, "D",
  "<p>Each time a question is asked, what does the Language Model actually read?</p>",
  [("The instruction, the few retrieved chunks, and the question", None),
   ("The whole document, plus the question",
    "That is Section 12's approach — chatting with a document by pasting the whole thing into the prompt. RAG exists so you do not have to."),
   ("Only the retrieved chunks",
    "The instruction and the question are in there too, or the model would not know what it is being asked or what rules to follow."),
   ("The whole vector store, since it has access to Chroma",
    "The model has no connection to the store. It sees only the text the flow hands it.")],
  "<p>That is the cost argument for RAG: no matter how big the document is, the model reads a page or two per question.</p>")

q(5, RAG, "C",
  "<p>You are upgrading the part 1 flow into a full RAG app. Which existing wire has to be <strong>removed</strong>?</p>",
  [("Chroma DB's <em>Search Results</em> → Chat Output", None),
   ("Chat Input → Chroma DB's <em>Search Query</em>",
    "Still needed. Every question still has to search the store."),
   ("The Embedding Model → Chroma DB's <em>Embedding</em>",
    "Still needed. Without it the question cannot be turned into a point to search with."),
   ("Split Text → Chroma DB's <em>Ingest Data</em>",
    "Still needed for loading. Generation changes what happens to the results, not how the document gets in.")],
  "<p>The results no longer go straight to the screen. Cut that wire and send Search Results into the Parser instead, so the chain ends at the model.</p>")

q(5, RAG, "A",
  "<p>In “retrieval-augmented generation,” what is being augmented, and with what?</p>",
  [("The model's answer is augmented with the chunks retrieved for that question", None),
   ("The model is retrained on the document before it answers",
    "Nothing about the model changes. RAG is prompting with better material, not training."),
   ("The model's context window is made larger to fit the document",
    "The window is fixed by the model. Retrieval is how you stay inside it."),
   ("The document is added to the store, which augments what the model knows",
    "Loading the store is the setup step. The augmenting happens at answer time, when retrieved text goes into the prompt.")],
  "<p>The model generates the answer; retrieval supplies the material it generates from. Same model, better-informed prompt.</p>")


# ================================================================== CHUNKING BY HAND
T = "Flows save automatically.\nName every flow clearly.\nKeep one flow per lab.\nDelete the test flows."
q(1, CHK, "C", "<p>Give the chunks and their lengths.</p>",
  [("Two chunks: 50 and 45", None),
   ("Two chunks: 49 and 44",
    "The separator was left out. Gluing two lines together costs one newline: 25 + 24 + 1 = 50."),
   ("Two chunks: 72 and 22",
    "Chunk 1 ran past the Chunk Size. 50 + 22 + 1 = 73 is over 60, so line 3 has to start a new chunk."),
   ("Four chunks: 25, 24, 22, 22",
    "The separator decides the atoms, not the chunks. Atoms keep getting glued together until the next one will not fit.")],
  "<p>Atoms: 25, 24, 22, 22.</p>"
  "<p>Buffer 25. Line 2: 25 + 24 + 1 = 50, not over 60 — it fits.</p>"
  "<p>Line 3: 50 + 22 + 1 = 73, over 60. <strong>Emit chunk 1 = 50.</strong> Overlap is 0, so the buffer empties. Buffer takes line 3: 22.</p>"
  "<p>Line 4: 22 + 22 + 1 = 45, fits. End of text: <strong>emit chunk 2 = 45.</strong></p>",
  doc=(T, N, 60, 0, [50, 45]))

T = "Open the canvas.\nDrag in Chat Input.\nConnect the model.\nRun the flow.\nRead the output."
q(1, CHK, "A", "<p>Which line appears in <strong>both</strong> chunk 1 and chunk 2?</p>",
  [("“Run the flow.”", None),
   ("“Connect the model.”",
    "It gets popped. After dropping lines 1 and 2 the buffer is 32 — still over 25 — so line 3 goes too."),
   ("“Read the output.”",
    "That line never makes it into chunk 1: 69 + 16 + 1 = 86 is over 70, so it starts chunk 2."),
   ("None — the chunks share no line",
    "Popping stops at 13, which is not over 25, so line 4 survives into chunk 2.")],
  "<p>Atoms: 16, 19, 18, 13, 16.</p>"
  "<p>Buffer: 16 → 36 → 55 → 69. Line 5: 69 + 16 + 1 = 86, over 70. <strong>Emit chunk 1 = 69</strong> (lines 1–4).</p>"
  "<p>Pop while over 25: 69 − 16 − 1 = 52, still over; 52 − 19 − 1 = 32, still over; 32 − 18 − 1 = 13. Stop. "
  "<strong>Line 4, “Run the flow.”, carries.</strong></p>"
  "<p>Buffer: 13 → 30. End: <strong>emit chunk 2 = 30.</strong></p>",
  doc=(T, N, 70, 25, [69, 30]))

T = "Langflow Desktop is free.\nThe canvas holds the flow.\nRun a component to test."
q(2, CHK, "D", "<p>How many chunks come out, and how long are they?</p>",
  [("One chunk of 77", None),
   ("Three chunks: 25, 26, 24",
    "Cutting is not chunking. The separator makes three atoms, but they all fit in one buffer, so they come out glued together."),
   ("One chunk of 75",
    "The two newlines that join the three lines are part of the chunk: 25 + 26 + 24 + 2 = 77."),
   ("One chunk of 1,000",
    "Chunk Size is the ceiling the buffer is tested against, not the length of the chunk. There are only 77 characters of document.")],
  "<p>Atoms: 25, 26, 24. Running totals: 25 → 52 → 77, and 77 is never over 1000.</p>"
  "<p>Nothing is ever emitted early, so at the end of the text the whole document comes out as <strong>one chunk of 77</strong>.</p>"
  "<p>A Chunk Size far bigger than the document does no cutting at all.</p>",
  doc=(T, N, 1000, 0, [77]))

T = ("Embeddings are points.\n\nThe store holds chunks and their points, and a search finds the "
     "points nearest to the question before any model ever sees the text.")
q(2, CHK, "B", "<p>How long is chunk 2?</p>",
  [("131 — the whole second paragraph, 71 characters over the Chunk Size", None),
   ("60 — Split Text cuts it off at the Chunk Size",
    "Nothing is ever cut mid-atom. Split Text only breaks the text at the separator, and there is no blank line inside that paragraph."),
   ("50 — the Chunk Size minus the overlap",
    "Overlap is text carried forward from the previous chunk, not an amount subtracted from the chunk."),
   ("There is no chunk 2 — the paragraph is dropped for being too long",
    "Nothing is ever discarded. An oversized atom comes out whole.")],
  "<p>With a blank line as the separator there are two atoms: 22 and 131.</p>"
  "<p>Buffer 22. The second atom: 22 + 131 + 2 = 155, over 60, so <strong>emit chunk 1 = 22</strong>. The buffer empties, then takes the big atom.</p>"
  "<p>End of text: <strong>emit chunk 2 = 131</strong>. Chunk Size is a target, not a cap — an atom bigger than the Chunk Size comes out in one piece.</p>",
  doc=(T, NN, 60, 10, [22, 131]), settings="Separator <code>\\n\\n</code> <span>Chunk Size <strong>60</strong></span> <span>Chunk Overlap <strong>10</strong></span>")

T = ("Load the document once.\nSearch it many times.\nWatch the token count.\n"
     "Keep the store small.\nAsk a hard question.\nCheck the answer.")
q(3, CHK, "C", "<p>Chunk Overlap is set to 20. How many lines do chunk 1 and chunk 2 share?</p>",
  [("None — the overlap comes out as zero", None),
   ("One line",
    "That would need the buffer to stop popping while a line was still in it. Every line here is longer than 20, so the last one standing gets popped too."),
   ("Two lines",
    "Two lines plus their newline would be more than 40 characters, far over the 20-character overlap budget."),
   ("Three lines",
    "The pop loop stops as soon as the buffer is no longer over 20, so it could never leave three lines behind.")],
  "<p>Atoms: 23, 21, 22, 21, 20, 17. Buffer: 23 → 45 → 68. Line 4: 68 + 21 + 1 = 90, over 80. <strong>Emit chunk 1 = 68.</strong></p>"
  "<p>Pop while over 20: 68 − 23 − 1 = 44, over; 44 − 21 − 1 = 22, over; 22 − 22 = 0. The buffer is empty.</p>"
  "<p>Overlap only survives when the lines are <em>smaller</em> than the overlap budget. Here every line is about the size of the budget itself, so <strong>nothing carries</strong>. The fix is the separator, not a bigger overlap number.</p>",
  doc=(T, N, 80, 20, [68, 60]))

T = ("Chat Input carries the question.\nChat Output shows the answer.\n\n"
     "The Parser makes plain text.\nThe prompt holds a blank.")
q(3, CHK, "A", "<p>Run this document twice with Chunk Size 80 and Chunk Overlap 50, once with Separator <code>\\n\\n</code> and once with <code>\\n</code>. What is the difference?</p>",
  [("<code>\\n\\n</code> gives 2 chunks that share nothing; <code>\\n</code> gives 3 chunks that each share a line", None),
   ("Both give the same chunks — the separator only changes where the text is cut",
    "Where the text is cut is exactly what decides the atom sizes, and atom sizes decide both the chunks and the overlap."),
   ("<code>\\n\\n</code> gives 2 chunks that share a paragraph; <code>\\n</code> gives 3 chunks that share nothing",
    "Backwards. The paragraph atoms (62 and 54) are bigger than the 50-character overlap, so they get popped and nothing carries."),
   ("<code>\\n\\n</code> gives 1 chunk, because the document has only one blank line",
    "One blank line makes two atoms, and 62 + 54 + 2 = 118 is over 80, so they cannot share a chunk.")],
  "<p>With <code>\\n\\n</code>: atoms 62 and 54. 62 + 54 + 2 = 118 is over 80, so <strong>chunks 62 and 54</strong>. Popping 62 empties the buffer, so no overlap.</p>"
  "<p>With <code>\\n</code>: atoms 32, 29, 28, 25 — small enough to carry. <strong>Chunks 62, 58 and 54</strong>, each sharing one line with the one before it.</p>"
  "<p>Same document, same numbers. The separator alone decided whether the overlap survived.</p>")

q(4, CHK, "B",
  "<p>A student sets Chunk Size to 1000 and gets chunks of 1,400, 2,100 and 900 characters. They are sure Langflow is broken. What actually happened, and what should they change?</p>",
  [("Their document has pieces longer than 1000 between separators; change the Separator", None),
   ("Chunk Size is being ignored; lower it to 500 and the chunks will shrink",
    "The 2,100-character piece has nowhere to be cut, so a smaller Chunk Size changes nothing about it. Only a separator that appears inside it can."),
   ("Chunk Overlap is being added on top; set the overlap to 0",
    "Overlap repeats text that is already in a chunk. It cannot push a chunk 1,100 characters past the target."),
   ("The document is too big for Split Text; split the file into smaller files first",
    "Document size is not the issue. One long unbroken paragraph would do this in a two-page file.")],
  "<p>Split Text cuts only at the separator, and it never cuts inside a piece. A paragraph with no separator inside it comes out whole, however long it is.</p>"
  "<p>With the default separator of <code>\\n</code> and unwrapped paragraphs, each paragraph is one atom. Switching to <code>. </code> makes sentence-sized atoms, and the chunks land near the target.</p>")

T = ("Chunk Size is a target.\nOverlap repeats text.\nSeparator cuts first.\nInspect the output.\n"
     "Run it again.\nName the collection.\nSearch the store.")
q(4, CHK, "D", "<p>How many chunks are there, and how long is the last one?</p>",
  [("Three chunks; the last is 52", None),
   ("Two chunks; the last is 52",
    "Counting the emits: one when line 5 will not fit, one when line 7 will not fit, and one at the end of the text — three in all."),
   ("Three chunks; the last is 37",
    "The carried line counts toward the last chunk: 13 + 20 + 1 + 17 + 1 = 52."),
   ("Four chunks; the last is 17",
    "That assumes each emit starts from an empty buffer. With Chunk Overlap 25 a line carries forward, which is why chunk 3 is longer than the lines left in it.")],
  "<p>Atoms: 23, 21, 21, 19, 13, 20, 17.</p>"
  "<p>Buffer 23 → 45 → 67. Line 4 would make 87, over 70: <strong>emit chunk 1 = 67</strong>. Pop while over 25 → 21 (line 3 carries).</p>"
  "<p>Buffer 21 → 41 → 55. Line 6 would make 76: <strong>emit chunk 2 = 55</strong>. Pop while over 25 → 13 (line 5 carries).</p>"
  "<p>Buffer 13 → 34 → 52. End of text: <strong>emit chunk 3 = 52</strong>.</p>",
  doc=(T, N, 70, 25, [67, 55, 52]))

q(5, CHK, "C",
  "<p>A student sets Chunk Overlap to 200 with Chunk Size 1000. No chunk is over 1000, but no two chunks share a single word. Why?</p>",
  [("Every atom is longer than 200, so the pop loop empties the buffer every time", None),
   ("Chunk Overlap only works when Chunk Size is a multiple of it",
    "There is no such rule. The two numbers are compared with the buffer, never with each other."),
   ("Overlap is applied only from chunk 2 onward, so it never shows up",
    "Overlap is carried after every emit, including the first one."),
   ("Keep Separator is off, which strips the repeated text",
    "Keep Separator decides whether the separator characters stay in the chunks. It has nothing to do with what carries forward.")],
  "<p>After emitting, atoms are popped off the front while the buffer is bigger than the overlap. If the last remaining atom is itself over 200, it gets popped too and the buffer empties.</p>"
  "<p>The fix is the separator: cut into smaller atoms (sentences instead of paragraphs) so something can fit under the overlap budget. Raising the overlap to 400 would also work, but it is the blunter move.</p>")

q(5, CHK, "A",
  "<p>A 40-page handbook: paragraphs of 3–6 sentences, separated by blank lines, hard-wrapped at about 80 characters a line. You want chunks near 800 characters that overlap by about a sentence. Which settings fit best?</p>",
  [("Separator <code>. </code>, Chunk Size 800, Chunk Overlap 150", None),
   ("Separator <code>\\n\\n</code>, Chunk Size 800, Chunk Overlap 150",
    "Paragraph atoms here run 400–900 characters. Some are bigger than the overlap budget, and some are bigger than the Chunk Size itself, so the overlap would often vanish."),
   ("Separator <code>\\n</code>, Chunk Size 800, Chunk Overlap 700",
    "An overlap that is most of the Chunk Size means chunks that are nearly all repeated text — far more chunks to embed for the same document."),
   ("Separator <code>. </code>, Chunk Size 150, Chunk Overlap 800",
    "An overlap bigger than the Chunk Size is rejected — Langflow will not run it.")],
  "<p>Sentences run about 100–150 characters here, so <code>. </code> gives atoms comfortably smaller than both the target and the overlap.</p>"
  "<p>An overlap of 150 is about one sentence: enough to carry, not so much that chunks are mostly repeats.</p>"
  "<p>Then check two things in Inspect output: is any chunk much longer than 800 (an atom too big), and do neighboring chunks actually repeat a sentence (if not, the atoms are over the overlap budget)?</p>")


# ================================================================== CHUNKING EDGE CASES
T = "Load the document.\nCheck the output.\nRead the chunks.\nSet the chunk size."
q(1, EDGE, "B", "<p>After chunk 1 is emitted, does “Check the output.” carry into chunk 2?</p>",
  [("Yes — after popping line 1 the buffer is exactly 17, which is not over 17", None),
   ("No — with its newline it counts as 18, which is over 17",
    "A buffer holding one atom holds no separator. The newline left with line 1 when line 1 was popped."),
   ("No — the pop loop always empties the buffer after an emit",
    "It stops as soon as the buffer is no longer bigger than the Chunk Overlap. Here it stops with line 2 still in it."),
   ("Yes — but only because Chunk Size 50 leaves room for it",
    "Room in the next chunk is a separate test. What decides whether it carries is the overlap test, and 17 is not over 17.")],
  "<p>Atoms: 18, 17, 16, 19. Buffer 18 → 36. Line 3 would make 53, over 50: <strong>emit chunk 1 = 36</strong>.</p>"
  "<p>Pop while over 17: drop line 1 <em>and the newline that joined it</em> → 36 − 18 − 1 = 17. Not over 17, so stop.</p>"
  "<p><strong>“Check the output.” carries</strong>, and it carries as 17 characters, not 18 — alone in the buffer, it has no separator attached.</p>",
  doc=(T, N, 50, 17, [36, 34, 36]))

T = "Name the collection.\nTry again.\nStop here.\nSearch the store nightly."
q(1, EDGE, "D", "<p>Chunk Overlap is 20 and the two short lines are 10 characters each. How many lines carry into chunk 2?</p>",
  [("One — “Stop here.” only", None),
   ("Two — 10 + 10 = 20, which is not over 20",
    "Both lines in the buffer together means the newline between them counts: 10 + 1 + 10 = 21, which <em>is</em> over 20, so the pop loop keeps going."),
   ("None — both short lines get popped",
    "Popping stops as soon as the buffer is no longer over 20. With one line left the buffer is 10, so it stops there."),
   ("Three — everything but the first line",
    "Line 1 is not the only thing over budget. The loop keeps popping while the buffer is over 20.")],
  "<p>Atoms: 20, 10, 10, 25. Buffer 20 → 31 → 42. Line 4 would make 68, over 50: <strong>emit chunk 1 = 42</strong>.</p>"
  "<p>Pop while over 20: drop line 1 → 42 − 20 − 1 = 21. Still over 20 (that 21 is 10 + 1 + 10), so drop line 2 → 10. Stop.</p>"
  "<p><strong>One line carries.</strong> A separator counts only between two atoms that are both in the buffer.</p>",
  doc=(T, N, 50, 20, [42, 36]))

T = "Load the document.\n\nCheck the output.\n\nRead the chunks.\n\nSet the chunk size."
q(2, EDGE, "C", "<p>The separator is <code>\\n\\n</code>. How many paragraphs carry from chunk 1 into chunk 2?</p>",
  [("One", None),
   ("Two",
    "That is what you get by counting each <code>\\n\\n</code> as one character: 18 + 1 + 16 = 35 would stop the popping. Counted properly it is 18 + 2 + 16 = 36, still over 34."),
   ("None",
    "Popping stops once the buffer is no longer over 34. With one paragraph left the buffer is 16."),
   ("Three",
    "All three would be 18 + 2 + 17 + 2 + 16 = 55, nowhere near the 34-character overlap budget.")],
  "<p>Atoms: 18, 17, 16, 19, with every separator worth <strong>2</strong> characters.</p>"
  "<p>Buffer 18 → 37 → 55. The last paragraph would make 76, over 60: <strong>emit chunk 1 = 55</strong>.</p>"
  "<p>Pop while over 34: 55 − 18 − 2 = 35, still over; 35 − 17 − 2 = 16. Stop. <strong>One paragraph carries.</strong></p>"
  "<p>Count that separator as 1 and you get 34, which is not over 34 — and you would wrongly carry two.</p>",
  doc=(T, NN, 60, 34, [55, 37]), settings="Separator <code>\\n\\n</code> <span>Chunk Size <strong>60</strong></span> <span>Chunk Overlap <strong>34</strong></span>")

T = "Check the output.\nRead the chunks.\nThe model reads only what the flow hands it."
q(2, EDGE, "A", "<p>Chunk Overlap is 25 and “Read the chunks.” is only 16 characters. Does it carry into chunk 2?</p>",
  [("No — keeping it would make chunk 2 overflow, and Chunk Size wins", None),
   ("Yes — 16 is not over 25, so the pop loop stops there",
    "The overlap test is satisfied, but the pop loop has a second condition: keep popping while the incoming atom still would not fit. 16 + 1 + 44 = 61 is over 60."),
   ("Yes — overlap is a promise that something always carries",
    "Overlap is a limit on how much <em>may</em> carry, not a guarantee that anything does."),
   ("No — an atom is never allowed to carry if the next atom is longer than it",
    "Length compared to the next atom is not a rule anywhere. What matters is whether the next atom fits beside what is left.")],
  "<p>Atoms: 17, 16, 44. Buffer 17 → 34. The long line would make 34 + 44 + 1 = 79, over 60: <strong>emit chunk 1 = 34</strong>.</p>"
  "<p>Pop while over 25: drop line 1 → 16. The overlap test is happy — but line 3 still does not fit (16 + 1 + 44 = 61 &gt; 60), so pop again → 0.</p>"
  "<p><strong>Chunk 2 = 44, with no overlap.</strong> Split Text will not keep an overlap at the cost of an oversized chunk.</p>",
  doc=(T, N, 60, 25, [34, 44]))

T = "Name the collection.\nRun the flow.\nWire the model.\nRetrieval returns something."
q(3, EDGE, "D", "<p>The overlap test would allow two lines to carry. How many actually do?</p>",
  [("One — Chunk Size only had room for one", None),
   ("Two — the overlap test is what decides",
    "It decides how much <em>may</em> carry. The incoming atom still has to fit beside it, and 29 + 1 + 28 = 58 is over 55."),
   ("None — once the incoming atom does not fit, the buffer empties",
    "The loop pops one atom at a time and stops as soon as the atom fits. Here it fits after one more pop."),
   ("Three — everything carries, since the buffer is under the Chunk Size",
    "The buffer is tested against the Chunk Overlap after an emit, not against the Chunk Size.")],
  "<p>Atoms: 20, 13, 15, 28. Buffer 20 → 34 → 50. Line 4 would make 79, over 55: <strong>emit chunk 1 = 50</strong>.</p>"
  "<p>Pop while over 30: drop line 1 → 29. Overlap test satisfied, with lines 2 and 3 still in the buffer.</p>"
  "<p>But does line 4 fit? 29 + 1 + 28 = 58, over 55 — so pop again → 15. Now 15 + 1 + 28 = 44 fits. <strong>One line carries</strong>, and chunk 2 = 44.</p>",
  doc=(T, N, 55, 30, [50, 44]))

T = "Check the output.\nSave the flow.\nAsk the app a hard question."
q(3, EDGE, "B", "<p>Chunk 2 would come to exactly 43 characters with a Chunk Size of 45. What happens?</p>",
  [("It fits — the test is “bigger than,” and 43 is not bigger than 45", None),
   ("It does not fit — a chunk has to stay safely under the Chunk Size",
    "There is no safety margin in the test. The comparison is strict: only a buffer <em>bigger</em> than the Chunk Size forces an emit."),
   ("It fits, but Split Text trims it to leave room for the separator",
    "The separator is already counted inside that 43 — it is the +1 in 14 + 1 + 28."),
   ("It does not fit, because the carried line no longer counts against the overlap",
    "Whether a line carries is settled before the next atom is added. Once it has carried, it is simply part of the buffer.")],
  "<p>Atoms: 17, 14, 28. Buffer 17 → 32. Line 3 would make 61, over 45: <strong>emit chunk 1 = 32</strong>.</p>"
  "<p>Pop while over 16: drop line 1 → 14. Stop. Line 3 fits beside it: 14 + 1 + 28 = 43, not over 45.</p>"
  "<p>End of text: <strong>emit chunk 2 = 43</strong>. Landing exactly on the Chunk Size would also be fine — every test here is strictly “bigger than.”</p>",
  doc=(T, N, 45, 16, [32, 43]))

T = "Name the collection.\nCheck the output.\nWatch the token count.\nRead every chunk."
q(4, EDGE, "C",
  "<p>A classmate writes: “Atoms 20, 17, 22, 17. Chunk 1 = 38. Pop to 17, line 2 carries. Chunk 2 = 40. Pop: drop line 2, leaving line 3. Line 3 plus its newline is 23, which is over 22, so drop it too. Chunk 3 = 17.” Where is the mistake?</p>",
  [("Line 3 is alone in the buffer, so it has no newline: the buffer is 22, not 23", None),
   ("The atoms are wrong — “Name the collection.” is 21 characters",
    "The newline is the separator; it is not part of the line it ends. The line itself is 20 characters."),
   ("Chunk 1 is wrong — it should be 37",
    "Two lines in the buffer means one newline between them: 20 + 1 + 17 = 38."),
   ("Popping line 2 is the mistake — the buffer was already under the overlap",
    "After chunk 2 is emitted the buffer is 40, which is over 22, so the popping does have to start.")],
  "<p>A separator counts only between two atoms that are both in the buffer. When line 2 is popped, the newline that joined it goes with it.</p>"
  "<p>So the buffer is 40 − 17 − 1 = <strong>22</strong>, which is not over 22. <strong>Line 3 carries</strong>, and chunk 3 = 22 + 1 + 17 = 40.</p>"
  "<p>The correct answer: three chunks — 38, 40 and 40 — each sharing one line with the one before it.</p>",
  doc=(T, N, 45, 22, [38, 40, 40]))

q(4, EDGE, "A",
  "<p>The buffer holds three lines of 12 characters each, joined by single newlines. Two lines are popped off the front. How big is the buffer now?</p>",
  [("12", None),
   ("13",
    "The newline that joined the last line to the one before it was popped along with that line. One atom alone carries no separator."),
   ("14",
    "That counts both of the popped separators as if they stayed behind. Each separator leaves with the atom it followed."),
   ("24",
    "That is two lines' worth. Three lines minus two leaves one.")],
  "<p>Three 12-character lines make a buffer of 12 + 1 + 12 + 1 + 12 = 38.</p>"
  "<p>Pop the first line: 38 − 12 − 1 = 25. Pop the second: 25 − 12 − 1 = <strong>12</strong>.</p>"
  "<p>A buffer holding one atom holds no separator. The newline that will join it to the next atom is counted when that atom arrives.</p>")

q(5, EDGE, "B",
  "<p>Which statement about Chunk Overlap is correct?</p>",
  [("It is a limit on how much may carry forward, not a promise that anything will", None),
   ("Every chunk after the first is guaranteed to repeat that many characters",
    "Nothing is guaranteed. If the last atom in the buffer is bigger than the overlap, the buffer empties and the overlap is zero."),
   ("It sets how much the chunks are allowed to exceed the Chunk Size",
    "Overlap never pushes a chunk past the Chunk Size — the fit test counts the carried text like any other text. Only an oversized atom does that."),
   ("It is measured in lines, so an overlap of 2 carries two lines",
    "Every one of these numbers is counted in characters, including the separators between atoms.")],
  "<p>After a chunk is emitted, atoms are popped off the front while the buffer is bigger than the overlap — and also while the next atom still would not fit.</p>"
  "<p>Whatever survives both tests is the overlap. Sometimes that is two lines, sometimes one, sometimes nothing at all.</p>")

q(5, EDGE, "D",
  "<p>Chunk Size 500, Chunk Overlap 100, Separator <code>\\n</code>. Every line in the document is about 300 characters long. What will the overlap be in practice?</p>",
  [("Zero — each line is bigger than 100, so the buffer always empties", None),
   ("100 characters, taken from the end of the previous chunk",
    "Split Text carries whole atoms, never a slice of one. A 300-character line cannot contribute a 100-character tail."),
   ("300 characters, since a whole line always carries",
    "A whole line carries only if it survives the pop loop, and 300 is over the 100-character budget."),
   ("It cannot be predicted without knowing how many lines there are",
    "The line length alone settles it: if every atom is bigger than the overlap, nothing can ever survive the pop loop.")],
  "<p>The pop loop keeps dropping atoms while the buffer is over 100. With 300-character atoms, even the last one standing is over the budget, so it goes too.</p>"
  "<p>To get a real overlap, cut into smaller atoms — change the separator to <code>. </code> — so that a full atom fits inside the overlap budget.</p>")


# ================================================================== PRECISION & RECALL
# precision = hits ÷ k, recall = hits ÷ relevant,
# best possible hits = min(k, relevant).
assert pct(3, 4) == 75 and pct(3, 6) == 50 and pct(4, 6) == 67
q(1, PR, "D",
  "<p><strong>Scenario.</strong> A handbook is split into 20 chunks. For the question “How do I book the laser cutter?” the answer key marks <strong>6 chunks relevant</strong> and 14 not relevant. You retrieve <strong>k = 4</strong> chunks and <strong>3</strong> of them are relevant. What is the <strong>precision</strong>?</p>",
  [("75%", None),
   ("50%",
    "That is 3 ÷ 6 — the hits divided by the relevant chunks. That is recall."),
   ("15%",
    "That is 3 ÷ 20, dividing by every chunk in the store. The store's size is in neither formula."),
   ("25%",
    "That is 1 ÷ 4, the share that was junk. Precision is the share that was useful.")],
  f"<p>Precision = hits ÷ k = {fr(3, 4)} = <strong>75%</strong>.</p>"
  "<p>Three of the four chunks you sent forward were worth sending.</p>")

q(1, PR, "C",
  "<p><strong>Same scenario</strong> (6 relevant, 14 not relevant, k = 4, 3 hits). What is the <strong>recall</strong>?</p>",
  [("50%", None),
   ("75%",
    "That is 3 ÷ 4 — the hits divided by k. That is precision."),
   ("21%",
    "That is 3 ÷ 14, dividing by the not-relevant count. That number is in no formula."),
   ("15%",
    "That is 3 ÷ 20, dividing by every chunk in the store.")],
  f"<p>Recall = hits ÷ relevant = {fr(3, 6)} = <strong>50%</strong>.</p>"
  "<p>Half of what could have answered the question actually came back.</p>")

q(2, PR, "A",
  "<p><strong>Same scenario</strong> (6 relevant, 14 not relevant, k = 4). What is the <strong>best possible recall</strong>?</p>",
  [("67%", None),
   ("100%",
    "A perfect retrieval still returns only 4 chunks, and there are 6 relevant ones. Some have to be left behind."),
   ("50%",
    "That is the recall actually achieved. The ceiling is what a perfect retrieval would have scored."),
   ("20%",
    "That is 4 ÷ 20, k divided by the whole store. The store's size is in no formula.")],
  f"<p>Best possible hits = the smaller of k and relevant = 4.</p>"
  f"<p>Best possible recall = {fr(4, 6)} = <strong>67%</strong>.</p>"
  "<p>So this run's 50% is not as bad as it looks — the ceiling was 67%, and it fell one chunk short.</p>")

assert pct(2, 10) == 20 and pct(2, 3) == 67 and pct(3, 10) == 30
q(2, PR, "B",
  "<p><strong>Scenario.</strong> For “When does the shop close?” the answer key marks <strong>3 chunks relevant</strong> and 17 not relevant. You retrieve <strong>k = 10</strong> and <strong>2</strong> are relevant. What is the <strong>best possible precision</strong>?</p>",
  [("30%", None),
   ("100%",
    "Only 3 chunks in the whole store are relevant, so 10 retrieved chunks must include at least 7 that are not."),
   ("20%",
    "That is the precision actually achieved, 2 ÷ 10. The ceiling asks what a perfect retrieval would have scored."),
   ("15%",
    "That is 3 ÷ 20, relevant divided by the store. k is what precision divides by.")],
  f"<p>Best possible hits = the smaller of k and relevant = 3.</p>"
  f"<p>Best possible precision = {fr(3, 10)} = <strong>30%</strong>.</p>"
  "<p>With k set to 10 and only 3 relevant chunks, low precision is built into the setting — it is not a sign the search went wrong.</p>")

q(3, PR, "C",
  "<p><strong>Same scenario</strong> (3 relevant, 17 not relevant, k = 10, 2 hits). What is the <strong>recall</strong>?</p>",
  [("67%", None),
   ("20%",
    "That is 2 ÷ 10 — the hits divided by k. That is precision."),
   ("10%",
    "That is 2 ÷ 20, dividing by every chunk in the store."),
   ("12%",
    "That is 2 ÷ 17, dividing by the not-relevant count, which is in no formula.")],
  f"<p>Recall = hits ÷ relevant = {fr(2, 3)} ≈ <strong>67%</strong>.</p>"
  "<p>Two of the three chunks that could have answered the question came back.</p>")

q(3, PR, "A",
  "<p>The answer key marks <strong>4 chunks relevant</strong> and 16 not relevant. Which Number of Results lets a perfect retrieval reach <strong>100% on both</strong> precision and recall?</p>",
  [("k = 4", None),
   ("k = 1",
    "Precision could be 100%, but one chunk out of four relevant ones caps recall at 25%."),
   ("k = 10",
    "Recall could be 100%, but 4 relevant chunks out of 10 retrieved caps precision at 40%."),
   ("k = 20",
    "Retrieving everything guarantees 100% recall and caps precision at 4 ÷ 20 = 20%.")],
  "<p>Best possible hits is the smaller of k and relevant. Both ceilings reach 100% only when those two numbers are equal.</p>"
  f"<p>With k = 4: best precision = {fr(4, 4)} = 100%, best recall = {fr(4, 4)} = 100%.</p>"
  "<p>Of course, different questions have different numbers of relevant chunks, so no single k is right for all of them.</p>")

q(4, PR, "D",
  "<p>You retrieve <strong>5</strong> chunks. Precision is <strong>60%</strong> and recall is <strong>50%</strong>. How many chunks does the answer key mark relevant?</p>",
  [("6", None),
   ("3",
    "That is the number of hits — the first step, not the answer. Recall then turns 3 into the relevant count."),
   ("5",
    "That is 3 ÷ 60%, which just recovers k. You already know k."),
   ("10",
    "That is 5 ÷ 50%, using k where the hits belong. Recall divides hits by relevant.")],
  f"<p>Precision = hits ÷ k, so 60% = hits ÷ 5 and hits = 3.</p>"
  "<p>Recall = hits ÷ relevant, so 50% = 3 ÷ relevant, which makes relevant = <strong>6</strong>.</p>")

q(4, PR, "B",
  "<p>The answer key marks <strong>4 chunks relevant</strong> and <strong>6 not relevant</strong> — 10 chunks in all. Which result could <strong>not</strong> happen?</p>",
  [("Retrieve 9 chunks, 2 of them relevant", None),
   ("Retrieve 6 chunks, none of them relevant",
    "Possible: there are exactly 6 not-relevant chunks, so a bad retrieval could return all six. Terrible, but possible."),
   ("Retrieve 10 chunks, 4 of them relevant",
    "Possible: retrieving everything always returns all 4 relevant chunks."),
   ("Retrieve 3 chunks, all 3 relevant",
    "Possible: there are 4 relevant chunks, so 3 of them is no problem.")],
  "<p>Retrieving 9 chunks with only 2 relevant would mean 7 not-relevant ones, but the store holds just 6.</p>"
  "<p>At least 9 − 6 = <strong>3</strong> of any 9 retrieved chunks must be relevant, so 2 hits is impossible.</p>"
  "<p>This is the one job of the not-relevant count: it is in no formula, but it tells you which results are even possible.</p>")

q(5, PR, "C",
  "<p>Team A used k = 10 and retrieved all 3 relevant chunks: precision 30%, recall 100%. Team B used k = 4 and retrieved 2 of them: precision 50%, recall 67%. Is it fair to call Team A's retrieval poor because its precision was only 30%?</p>",
  [("No — with k = 10 and 3 relevant chunks, 30% <em>is</em> the best possible precision", None),
   ("Yes — most of what Team A retrieved was junk",
    "The junk was forced by the setting, not by the search. No retrieval with k = 10 could have done better than 30%."),
   ("Yes — Team B's precision was higher, so Team B retrieved better",
    "Raw scores from different values of k are not comparable. Team B fell short of both of its ceilings; Team A hit both of its own."),
   ("No — Team A's real precision was 3 ÷ 3 = 100%",
    "That is Team A's recall. Precision always divides by k.")],
  "<p>Team A's ceilings: precision 3 ÷ 10 = 30%, recall 3 ÷ 3 = 100%. It hit both — for that setting, retrieval was perfect.</p>"
  "<p>Team B's ceilings: precision 3 ÷ 4 = 75%, recall 100%. It scored 50% and 67%, short on both, even with room for every relevant chunk.</p>"
  "<p>The fair complaint about Team A is about the setting: k = 10 sends 7 useless chunks to the model every question, and you pay for those tokens. That is a reason to lower k, not evidence that the search failed.</p>")

q(5, PR, "A",
  "<p>A question has <strong>2 relevant chunks</strong> in the store. You retrieve <strong>k = 5</strong> and get both of them. What are the precision and recall?</p>",
  [("Precision 40%, recall 100%", None),
   ("Precision 100%, recall 40%",
    "Swapped denominators. Precision divides by k = 5; recall divides by relevant = 2."),
   ("Precision 40%, recall 40%",
    "Recall divides by the relevant count, not by k: 2 ÷ 2 = 100%."),
   ("Precision 100%, recall 100%",
    "Perfect recall, yes — but three of the five retrieved chunks were not relevant, so precision is not 100%.")],
  f"<p>Precision = {fr(2, 5)} = <strong>40%</strong>. Recall = {fr(2, 2)} = <strong>100%</strong>.</p>"
  "<p>This is the usual shape of a high-k setting: every relevant chunk is found, and plenty of filler comes with it.</p>")


# ------------------------------------------------------------------ assemble + check
LETTERS = "ABCD"
tests = {t: [] for t in range(1, 6)}
letter_count = {L: 0 for L in LETTERS}

for item in Q:
    ch = item["choices"]
    assert len(ch) == 4, item["prompt"]
    assert ch[0][1] is None and all(w for _, w in ch[1:]), item["prompt"]
    assert len({c for c, _ in ch}) == 4, "duplicate choice: " + item["prompt"]
    k = LETTERS.index(item["pos"])
    ordered = ch[1:k + 1] + [ch[0]] + ch[k + 1:]
    letter_count[item["pos"]] += 1
    out = dict(topic=item["topic"], prompt=item["prompt"],
               choices=[c for c, _ in ordered], answer=k,
               why=[w for _, w in ordered], solution=item["solution"])
    if item["doc"]:
        text, sep, size, overlap, expected = item["doc"]
        check_chunks(text, sep, size, overlap, expected)
        out["doc"] = dict(text=text, sep=sep, size=size, overlap=overlap)
        if item["settings"]:
            out["settings"] = item["settings"]
    tests[item["test"]].append(out)

for t, qs in tests.items():
    assert len(qs) == 10, f"test {t} has {len(qs)} questions"
    topics = [x["topic"] for x in qs]
    assert all(topics.count(tp) == 2 for tp in (CHK, EDGE, PR, LOAD, RAG)), f"test {t} topic mix: {topics}"

assert max(letter_count.values()) - min(letter_count.values()) <= 1, letter_count

data = [dict(number=t, questions=qs) for t, qs in tests.items()]
with open(OUT, "w", encoding="utf-8") as f:
    f.write("/* GENERATED by tools/build_questions.py — do not edit by hand. */\n")
    f.write("window.PRACTICE_TESTS = ")
    json.dump(data, f, ensure_ascii=False, indent=1)
    f.write(";\n")

print(f"OK: {len(Q)} questions, 5 tests, answer letters {letter_count}, "
      f"chunking checked against {'REAL langchain splitter' if HAVE_LANGCHAIN else 'built-in algorithm only'}")
print("wrote", os.path.normpath(OUT))

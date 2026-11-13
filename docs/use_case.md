# Use case: finding evidence in your papers

This guide explains who Research Radar is for, what each step does, and what
kinds of questions work well. To install and start it, see the
[README](../README.md).

## Why use it

You have a handful of papers (PDFs) and specific questions: which dataset did
they use, how many patients, how was something defined, what are the
limitations? Reading every paper in full takes hours. Searching for a word
only finds that exact word. A general chatbot answers fluently but may invent
facts or citations.

Research Radar answers **only from your PDFs**. Every answer shows the passage
and the page it came from, so you can check it. When your papers do not
contain the answer, it says so instead of guessing. Everything runs on your
own computer.

**A good fit:** screening papers before reading them closely, finding a number,
definition or method quickly, and checking a claim before you cite it.

**Not a fit yet (version 1):** summaries of a whole paper, automatic
comparisons between papers, scanned PDFs (no OCR), and follow-up questions that
depend on earlier answers. See the [roadmap](roadmap.md).

## Step by step

### 1. Upload your papers
In the **Research Assistant** page, use **Upload PDFs** in the sidebar and
pick one or more files. They must contain real text (not scans) and be at most
20 MB each.

### 2. Click "Index papers"
Uploading only selects the files. **Index papers** is the step that makes them
searchable. For each paper it:

1. reads the text page by page and remembers the page numbers;
2. cleans the text (extra spaces, broken lines, hyphenation);
3. splits it into passages of about 1,200 characters, slightly overlapping;
4. turns each passage into a list of 768 numbers that captures its meaning
   (an embedding, made by the EmbeddingGemma model running locally);
5. stores the passages, their numbers, the file name and the page in a local
   database on your computer, and records the paper in a list.

This is done **once per paper**. Three papers take about a minute. Afterwards
the paper stays in the sidebar, survives restarts, and questions are fast. The
trash icon next to a paper deletes it.

### 3. Choose which papers to search
The checkboxes in the sidebar decide which papers a question searches. Untick
papers that are not relevant, especially when several papers could match.

### 4. Ask a question
When you press enter, the app:

1. turns your question into numbers with the same model;
2. finds the passages with the closest meaning (5 by default; the slider
   **Passages to retrieve** sets 1 to 10);
3. gives those passages to a local language model (Qwen3) with the rule to
   answer only from them and to cite each statement as `[n]`;
4. checks every citation in code: invented numbers are dropped, and an answer
   with no valid citation is replaced by "not enough evidence".

The first answer after a pause is slower because the model has to load.

### 5. Read the answer and check the evidence
- The numbers `[1]`, `[2]` in the answer match the numbered sources listed
  under it. Open one to see the file, the page and the **original text**.
- **All retrieved passages** shows everything that was found, cited or not,
  with its distance (lower means closer to your question).
- Always read the cited text for claims that matter. The check only proves that
  a cited passage exists, not that the claim is correct: in the project's
  evaluation, the slower model once gave a wrong answer while citing a correct
  page. For an important claim, open the PDF at that page.

### 6. When it says there is not enough evidence
The passages found did not support an answer. Either the papers really do not
contain it, or your wording did not match the paper's. Try the paper's own
terms, a more specific question, or more passages. This reply is deliberate:
the tool prefers saying nothing over inventing evidence.

## What to ask

| Kind of question | Example | Works well? |
|---|---|---|
| A fact or number | How many patients and how many centers does the registry include? | Yes |
| A definition | How was significant coronary stenosis defined? | Yes |
| A method detail | Which data augmentation techniques were applied? | Yes |
| A result | How accurate was the AI system compared with cardiologists? | Yes |
| Limitations | What limitations does the registry have? | Yes |
| Spread over many pages | What are the main types of MR artifacts in the scanner? | Less reliably |
| Whole-paper summary | Summarise this paper | Not supported |
| Opinion across papers | Which paper is better? | Not supported |
| Content of a figure | What does Figure 3 show? | Not supported (images are not read, only the text) |

Tips:

- Use the words the paper uses and ask **one thing** per question.
- Every question is answered **on its own**. Include the context in each
  question ("What were the limitations of the REFINE SPECT registry?" and not
  "And its limitations?").
- Tables are read as plain text and can be hard to interpret.
- If a broad question misses part of the answer, raise **Passages to
  retrieve**. In the benchmark, Recall@10 was 0.96 against 0.89 at 5.

## An example session

Using the three open-access papers from the project's benchmark (sources in
`evaluation/datasets/papers.md`): two about heart imaging with SPECT and one
about EEG recorded during fMRI. Wording of answers can vary slightly.

1. *How many patients and how many centers does the updated REFINE SPECT
   registry include?* gives 45,252 patients from 13 centers, with a citation.
2. *How accurate was the AI system compared with cardiologists working with and
   without AI guidance?* gives about 88% for the AI, 80% for cardiologists with
   its guidance and 65% without.
3. *Which probabilistic programming language was used to estimate the
   statistical models?* gives Stan, from the EEG paper. With only the two SPECT
   papers ticked, the same question gets "not enough evidence".
4. *What is the recommended dose of beta blockers for patients with coronary
   artery disease?* gets "not enough evidence": the papers do not say.

## Checking how well it works

The **Evaluation** page shows how the system was measured on these papers:
how often the right pages are found, which questions failed, and how two
language models compare. See [experiments](experiments.md) for the results and
their limits.

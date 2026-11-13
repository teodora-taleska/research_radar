# Demo video script

A 4 to 5 minute screen recording of one use case: **a researcher needs facts
from three papers quickly, and wants proof for every claim.** Every question
below was run against the real application, and the answers shown are what it
produced (wording can vary slightly between runs).

## Before recording

Preparation (about 10 minutes):

1. **Free memory.** Close Firefox tabs, Slack, Discord, Spotify and anything
   heavy. The language model needs about 3 GB while it answers.
2. **Use the native version** (two terminals), not Docker, unless you want to
   show Docker separately:
   - Terminal 1: `uvicorn app.api.main:app`
   - Terminal 2: `streamlit run frontend/Home.py`
3. **Start from an empty app.** Stop both, delete the `data` folder, start
   again. Or use the trash button next to each paper in the sidebar.
4. **Warm up the model.** Upload one paper, ask one question, then delete the
   paper. The first answer after a pause is slow because the model loads.
5. **Have the three PDFs ready** (see `evaluation/datasets/papers.md`):
   the two SPECT papers and the EEG-fMRI paper.
6. **Recording setup:** 1920 x 1080, browser zoom 110 to 125 percent, hide the
   bookmarks bar, turn on Windows Focus Assist so notifications stay quiet.
   Tools: Xbox Game Bar (Win + G) or OBS Studio.
7. **Open in tabs:** the app (http://localhost:8501), the architecture diagram
   (`docs/images/data-flows.svg` in a browser) and one of the PDFs.

## Storyboard

| Time | On screen | What to say | What to do |
|---|---|---|---|
| 0:00 | Home page | "Reading papers takes hours. Research Radar lets me ask questions about my own PDFs and get answers with the page as proof. It runs entirely on my computer: no cloud, no API keys." | Show the title and tagline |
| 0:20 | `data-flows.svg` | "A paper is split into passages, turned into vectors and stored. A question finds the closest passages, a local language model writes the answer from them only, and the system checks every citation." | Point at flows A and B for about 20 seconds |
| 0:45 | Research Assistant | "I upload three papers: two about AI for heart imaging and one about EEG." | Upload the 3 PDFs, click **Index papers**. Indexing takes about a minute: speed it up when editing |
| 1:15 | Sidebar | "Each paper is listed with its pages and passages." | Hover the help icon on a paper |
| 1:25 | Question 1 | "First a factual question." | Ask: **How many patients and how many centers does the updated REFINE SPECT registry include?** Expected: 45,252 patients from 13 centers, with a citation. Open the citation: "this is the original text, with the page number" |
| 1:55 | Question 2 | "Now a comparison across a result." | Ask: **How accurate was the AI system compared with cardiologists working with and without AI guidance?** Expected: 88 percent for the AI, 80 percent for cardiologists with AI, 65 percent without. Open both citations |
| 2:30 | Question 3 | "A summary question." | Ask: **What limitations does the REFINE SPECT registry have?** Expected: visual stenosis assessment, small subgroups, differences between centers |
| 2:55 | Paper selection | "I can choose which papers to search." | Ask: **Which probabilistic programming language was used to estimate the statistical models?** Expected: Stan. Then untick the EEG paper and ask again: it now says there is not enough evidence |
| 3:25 | Honest refusal | "When the papers do not contain the answer, it says so instead of guessing." | Tick all papers again. Ask: **What is the recommended dose of beta blockers for patients with coronary artery disease?** Expected: "The uploaded documents do not contain enough evidence". Open **All retrieved passages** to show what it found |
| 3:50 | Evaluation, Retrieval tab | "I measured the system on a benchmark: three papers and 26 questions with the relevant pages labelled. Recall at 5 is 0.89: for most questions the right pages are among the five passages the model sees." | Show the metrics and the chart, then open one failure case (for example q17) to show the missed pages |
| 4:20 | Evaluation, LLM comparison tab | "I compared two local models. They behave the same on the automatic checks, but the instruct model answers in 2.4 seconds instead of 22. One answer from the slower model was wrong while still citing a correct page, which is why I also score faithfulness by hand." | Show the summary table and the speed chart, then the answers for question q20 side by side |
| 4:50 | Close | "222 automated tests, Docker support, and a clear list of limitations. Next version: paper summaries, paper comparison and saved conversations." | Show the README roadmap, then the GitHub link |

## Tips

- Say the numbers once and keep sentences short. About 150 words a minute is
  comfortable.
- Record the narration live, or record the screen first and add the voice
  afterwards; both work.
- If an answer takes long, keep talking about what is happening ("it is
  retrieving the passages and generating the answer locally"), or cut the wait
  when editing.
- If an answer is worded differently from the table, that is normal. The
  numbers and the citations should match.
- Do not skip the refusal: it is the feature that shows the system does not
  invent evidence.

## After recording

1. Add the video link to the **Demo** section of `README.md`.
2. Take screenshots of the Research Assistant (with a cited answer) and of the
   Evaluation page, save them in `docs/images/`, and add them to the README.
3. Export the diagrams for slides if needed: `docs/images/architecture.png`
   and `docs/images/data-flows.png` are included next to the SVG files.

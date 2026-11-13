# Benchmark papers

The PDFs are not committed. Download them from the publishers and place them
in `evaluation/datasets/papers/` with these file names.

| File | Paper | DOI |
|---|---|---|
| `jnumed.124.268292.pdf` | Miller et al., The Updated Registry of Fast Myocardial Perfusion Imaging with Next-Generation SPECT (REFINE SPECT 2.0), J Nucl Med 2024 | 10.2967/jnumed.124.268292 |
| `s41598-024-64445-2.pdf` | Kusumoto et al., A deep learning-based automated diagnosis system for SPECT myocardial perfusion imaging, Scientific Reports 2024 | 10.1038/s41598-024-64445-2 |
| `fnimg-01-968363.pdf` | Kraljic et al., Evaluation and comparison of most prevalent artifact reduction methods for EEG acquired simultaneously with fMRI, Frontiers in Neuroimaging 2022 | 10.3389/fnimg.2022.968363 |

Two papers cover the same field (SPECT myocardial perfusion imaging) so that
retrieval has to tell similar documents apart. The EEG-fMRI paper comes from
a different field and acts as a distractor. All three are text-based PDFs
(no OCR needed).

Page numbers in `questions.json` are the PDF page indexes (page 1 is the
first page of the PDF file), which differ from the journal page numbers.

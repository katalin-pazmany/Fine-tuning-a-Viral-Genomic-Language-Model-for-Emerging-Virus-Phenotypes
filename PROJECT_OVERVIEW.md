# Project Overview — in plain English

*What this project does, why we did each step, and where every piece of data comes from.*
*Written as a plain-language companion to the Methods and Results chapters.*

---

## 1. The big question

Can an artificial-intelligence model that "reads" viral genomes predict dangerous traits of a virus
from its genetic sequence alone?

The specific model is **Vir2vec** — a large "genomic language model" (the same kind of technology as a
text language model, but trained on DNA instead of words). It has already been trained on hundreds of
thousands of viral genomes, and it turns any genome into a list of 768 numbers (an "embedding") that is
meant to capture what kind of virus it is. The project asks whether those numbers are useful for
predicting three risky viral behaviours, and — crucially — whether the expensive AI is actually better
than simple, old-fashioned methods.

## 2. The three things we predict

Each is a yes/no question about a virus:

1. **Human-to-human transmission** — can the virus spread directly from person to person?
2. **Zoonotic** — does the virus live in *both* animals and humans (so it can cross between them)? Note
   this is "animal ⇄ human", not specifically mammals.
3. **Vector-borne** — for a virus carried by an arthropod (mosquito, tick, or biting midge), does it
   break out of the bug and infect vertebrates (in practice: humans)?

## 3. Where all the data comes from

There are four kinds of data: the **labels** (the yes/no answers we train against), the **sequences**
(the genomes themselves), the **AI model**, and some **external datasets** used for cross-checking.

### 3a. The labels (the "right answers")

- **Human-to-human & zoonotic labels** come from a curated reference set of human-associated viruses,
  `data/human_pathogens.xlsx`, combined with virus–host association records from the **VIRION** database
  (`data/virion_edgelist.csv`, `data/virion_phenotype_labels.csv`). Human-to-human is a direct expert
  annotation. Zoonotic is derived: a virus is zoonotic if it is recorded infecting humans *and* at least
  one non-human host.
- Because the raw zoonotic label is noisy (some "animal hosts" are really just lab animals), the
  supervisor (**Maya**) reviewed the list and produced a **strict** version that removes those
  artefacts. Both the relaxed and strict versions are carried through everything, so every result can be
  checked against both. The final combined label file is
  `data/phenotype_labels_with_strict_zoonotic.csv` (Maya's reviewed inputs are
  `data/complete_h2h_labels.xlsx` and `data/complete_zoonotic_labels.xlsx`).
- **Vector-borne labels** (`data/vector_borne_labels.csv`) are built in two parts:
  - Mosquito- and tick-associated viruses come from the **ZOVER** database (an arbovirus / tick-virus
    catalogue).
  - Biting-midge viruses are *not* in ZOVER, so they were pulled directly from **NCBI** and
    **host-verified**: only records whose GenBank `/host=` field confirms a *Culicoides* biting midge
    were kept (this avoids false matches from loose text searches). That added **112 unique midge taxa**
    from 1,101 verified sequences.
  - A vector-associated virus is then labelled "positive" if it also appears in the human-infecting
    reference set — i.e. it crosses out of the arthropod into humans.

### 3b. The sequences (the genomes)

- Genome sequences were downloaded from **NCBI** (the public sequence database) as one FASTA file per
  accession, living under `data/raw_matched/` and the ZOVER / *Culicoides* sequence folders.
- Each sequence's accession was mapped to an NCBI taxonomy ID and cached in
  `outputs/accession_taxid_cache.csv`, so each virus (taxon) can be tied to its genomes.

### 3c. The AI model (Vir2vec)

- **Vir2vec** is a pre-trained viral genomic language model (422 million parameters), downloaded from
  Hugging Face (`pabloarozarenad/Vir2vec`). We did **not** train it — we use it as-is to convert genomes
  into embeddings.
- We ran every one of our genomes through Vir2vec once and saved the resulting embeddings to
  `outputs/refseq_embeddings.npz` (with `outputs/refseq_metadata.csv`), so the expensive step is done
  only once and reused everywhere.
- Importantly for the headline finding, Vir2vec's authors **published the exact list of genomes it was
  trained on** (on their GitHub, `simoRancati/Vir2vec`): 565,747 genomes across ~295 species, from
  BV-BRC, NCBI Virus, GISAID, HBVdb and the LANL HIV database. We downloaded those lists to work out
  which of our viruses the model had already seen (see step 4h).

### 3d. External datasets for cross-checking

- **Mollentze et al. (2021)** — a well-known published dataset and method for sequence-based zoonotic
  risk (from the `nardus/zoonotic_rank` repository). Used as a gold-standard comparison.
- **IntAct** (EBI) — a database of documented protein–protein interactions, queried live over its
  PSICQUIC web service. Used as an independent biological check. (The originally planned database,
  HPIDB, was offline, so IntAct was substituted.)
- **NCBI taxonomy dump** — the official taxonomy tree, used to match viruses at the species level.

### 3e. Where the computing happened

All the heavy work ran on the **University of Liverpool "Barkla" supercomputer**. The frozen-embedding
and baseline analyses run on ordinary processors; the fine-tuning experiments need graphics cards
(GPUs). Everything uses a fixed random seed so results are reproducible.

## 4. What we did, step by step — and why

### 4a. Clean the data to prevent cheating (deduplication)

Many viruses are represented by near-identical strains. If a near-copy of a virus ends up in both the
training and test sets, the model can "memorise" rather than genuinely predict, and the scores look
better than they really are. So before anything else, we clustered the genomes at 90% similarity
(**CD-HIT**) and kept one representative per cluster. This removed about **21%** of the
human/zoonotic viruses and **18%** of the vector-borne ones — proof the leakage risk was real.

### 4b. Turn genomes into numbers (frozen embeddings)

Every genome was run through Vir2vec once to get its 768-number embedding. "Frozen" means we don't
change the model — we just read out its representation.

### 4c. Three modelling strategies ("tiers"), to see if fine-tuning helps

- **Tier 1** — freeze Vir2vec, train a simple classifier on its embeddings.
- **Tier 2** — unfreeze the top couple of layers and fine-tune them a little.
- **Tier 3** — fine-tune using LoRA (a lightweight adaptation method).
Why: the natural assumption is that adapting the model more should help. We test that assumption.

### 4d. Simple baselines, to check the AI is worth it

We built deliberately cheap methods and made the AI compete against them:
- **Majority-class** (always guess the common answer) — the absolute floor.
- **GC-content + genome length** (just two numbers per genome).
- **k-mer composition** (counting short DNA "words").
- **Codon usage** (how the genome uses the 64 codons — a biology-flavoured composition measure).
- **BLAST** (classic sequence matching: copy the label of the most similar known virus).
Why: if a two-number or word-counting method does as well as a 422-million-parameter model, the model
isn't learning special biology — it's an expensive way to measure composition.

### 4e. Fair thresholds and honest metrics

Because some tasks are imbalanced (few positives), plain accuracy is misleading. We pick decision
thresholds properly (Youden's method) and report several metrics — AUROC, AUPRC, MCC — not just
accuracy.

### 4f. "Unseen family" tests (structured hold-outs)

We removed an entire virus family from training and tested on it, to see if methods generalise to
viruses unlike anything they trained on. We used Flaviviridae for human-to-human, and added
**Adenoviridae** for zoonotic (because Flaviviridae are almost all zoonotic, which made that test
meaningless). For vector-borne, we held out all tick-only viruses.

### 4g. External and biological cross-checks

- Compared Vir2vec against the **Mollentze** published benchmark.
- Checked whether high-risk predictions line up with documented human-protein interactions in
  **IntAct** (an independent signal that doesn't use our labels).
- Looked at *where* the models fail (misclassification by virus family / host group).
- Ran the vector-borne task with and without biting midges, to confirm adding them didn't change the
  conclusions.

### 4h. The key test: viruses the model never saw

Maya pointed out the deepest problem: Vir2vec was pre-trained on so many viruses that even a "held-out"
family may already be baked into its embeddings. So we used Vir2vec's **published training list** to
label each of our viruses as **seen** or **unseen** by the model (matched at species level using the
NCBI taxonomy). About half our viruses were genuinely unseen. We then did the honest, real-world test:
**train only on the viruses the model knew, and predict the ones it never saw** — the true
"emerging virus" scenario. Finally, we bootstrapped the results to get confidence intervals and check
the differences are statistically real, not noise.

## 5. What we found

1. **The simplest strategy wins.** Frozen embeddings with a basic classifier beat both fine-tuning
   approaches, on every task. Fine-tuning did not pay for itself.
2. **The leakage control mattered and the result survived it.** De-duplication lowered scores slightly
   but didn't change the story.
3. **On ordinary tests the cheap baselines are already competitive** — k-mer and codon usage roughly
   tie the AI, and even a two-number GC/length model is well above chance.
4. **On genuinely novel viruses, the AI has no advantage.** In the honest "train on known, predict
   unknown" test, Vir2vec matches or *loses* to the simple baselines on every task — and for
   human-to-human transmission it is *significantly* beaten by codon usage and k-mer counting
   (statistically confirmed). Its value is interpolation within familiar viral diversity, not
   extrapolation to the emerging viruses that actually matter.
5. **Honest negatives:** Vir2vec does not beat the published Mollentze method, and the IntAct check came
   back inconclusive (a non-significant null). Both are reported as they are.

**The one-sentence takeaway:** the pre-trained viral language model is good at recognising *what kind of
virus* something is, and phenotype predictions ride along with that — but for anticipating genuinely new
viruses it offers no advantage over simple, cheap, interpretable methods.

## 6. Where to find things (quick map)

- Labels: `data/phenotype_labels_with_strict_zoonotic.csv`, `data/vector_borne_labels.csv`
- Sequences: `data/raw_matched/` and the ZOVER / *Culicoides* sequence folders; accession→taxid cache
  `outputs/accession_taxid_cache.csv`
- Embeddings: `outputs/refseq_embeddings.npz`
- De-duplication: `outputs/dedup/`
- Which viruses Vir2vec saw: `outputs/vir2vec_seen_labels.csv` (built by
  `scripts/make_vir2vec_seen_labels.py`)
- Main results table: `outputs/results_summary_final_dedup.csv`
- The unseen-virus analysis: `scripts/seen_unseen_analysis.py`,
  `scripts/seen_unseen_significance.py`
- Write-up: `dissertation_docs/Methods.md`, `dissertation_docs/Results.md`,
  `dissertation_docs/Appendix.md`

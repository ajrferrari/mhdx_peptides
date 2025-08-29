# mhdx_peptides

Tools for analysis of mhdx on digested samples

### Summary of how we learn protease rules (from a non-specific FragPipe run)

We first map all confidently observed peptides back to the input protein sequences and treat each peptide boundary as an empirical cleavage site.
For the **C-terminus analysis**, we tabulate the residue at **P1** (the last residue of the upstream fragment) across all observed C-boundaries and compare its *observed proportion* to a **background** composition computed from internal positions of the relevant protein set (by default, only proteins with ≥1 identified peptide to avoid bias). The **enrichment** for residue \(a\) is


enrichment(a) = P(P1 = a in observed boundaries) / P(internal = a in background)



so values >1 indicate preference. For each residue we also compute a **Fisher’s exact test** against the background (two-sided for C-terminus tables), and convert the resulting p-values to **q-values** using Benjamini–Hochberg FDR control; residues passing a chosen enrichment threshold and \(q\)-value cutoff form the rule sets (e.g., *strict/balanced/broad*). In parallel, we report **P1′** (the first residue of the downstream fragment) as a diagnostic of sequence context (often showing proline effects) but we do not include P1′ when defining the C-rule.

To test for **N-terminal preferences independent of the C-rule**, we scan each amino acid \(X\) at peptide starts and ask whether starts with \(X\) remain frequent **even when the upstream residue is not in the learned C-rule set** \(S\).
We compare

P(prev ∉ S | start = X) in the observed peptides to P(prev ∉ S | curr = X)

in the background to obtain an enrichment and a one-sided Fisher p-value (BH-adjusted to a q-value).

**Plots**:
- *Boundary composition*: raw percentages for C-P1, N-P1, and background (y-axis is %), showing relative abundance.
- *Enrichment plots*: bars or lollipops showing obs/bg fold-change with reference line at 1, highlighting specificity.
- *N-scan scatter*: each residue shown by enrichment vs. observed conditional probability, point size reflecting evidence.

Residues with **high enrichment and low q-value** are interpreted as genuine preferences and are candidates for inclusion in the final regex rules (e.g., `(?<=[LFE...])` for C-cuts, optionally accompanied by a separate N-rule like `(?=[IM...])` if supported).


### Peptide Mapping Pipeline

The `run_pipeline` module automates the comparison of *theoretical* peptide digests to *experimentally observed* peptides (e.g. FragPipe output).

1. **In-silico digestion**
   Protein sequences are digested using a user-specified cleavage rule (e.g. Pepsin, Nep2, Trypsin) with configurable parameters for maximum missed cleavages, minimum/maximum peptide length, and whether digestion is fully enzymatic or semi-specific. All possible peptides are bucketed by missed cleavage count, and monoisotopic masses are calculated.

2. **Uniqueness annotation**
   Each theoretical peptide is annotated for uniqueness:
   - **By sequence** (default): a peptide is unique if it maps to only one protein in the dataset.
   - **By mass** (optional): peptides are compared within a ppm tolerance to identify potential ambiguities.

3. **Experimental peptide mapping**
   Observed peptides from FragPipe are cleaned (removing modifications, symbols) and mapped back onto the protein sequences. These are then matched to the theoretical digest to mark which peptides were detected experimentally.

4. **Visualization**
   - A **stacked barplot** summarizes unique vs. non-unique peptide counts for all proteins.
   - For each protein, a **peptide map** is generated showing the amino acid sequence with rectangles marking theoretical unique peptides (left panel) and experimentally observed peptides (right panel), aligned for easy comparison. Proteins are sorted by the number of unique observed peptides.

5. **Output organization**
   Results are saved into an output folder with two subdirectories:
   - `dataframe/` → JSON file containing the full `df_unique` table with theoretical peptides, uniqueness, and observed status.
   - `figs/` → PNG figures for the summary barplot and per-protein peptide maps.

This pipeline makes it straightforward to evaluate experimental coverage relative to theoretical digests, visualize which peptides were uniquely detected, and assess digestion efficiency across proteins.

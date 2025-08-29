# ---------- rule_learning.py (or paste into tools.py) ----------
import re
from typing import Iterable, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_SET = set(AA20)

# -------------------- Background helpers --------------------
def _bg_protein_subset(df_i: pd.DataFrame,
                       df_obs_map: pd.DataFrame,
                       bg_scope: str = "identified") -> pd.DataFrame:
    """Choose proteins to form the background."""
    if bg_scope not in {"identified", "all"}:
        raise ValueError("bg_scope must be 'identified' or 'all'")
    if bg_scope == "identified":
        names = set(df_obs_map["name"].unique())
        return df_i[df_i["name"].isin(names)].copy()
    return df_i.copy()

def _bg_internal_freq(df_bg: pd.DataFrame) -> pd.Series:
    """Background frequencies over internal positions only."""
    counts = pd.Series(0, index=AA20, dtype=int)
    total = 0
    for _, r in df_bg.iterrows():
        s = str(r["sequence"]).upper()
        if len(s) <= 2:
            continue
        for ch in s[1:-1]:  # internal only
            if ch in AA20_SET:
                counts[ch] += 1
                total += 1
    if total == 0:
        raise ValueError("Background total is zero; check sequences/bg_scope.")
    return (counts / total).rename("bg_freq").replace(0, 1e-9)

# -------------------- Boundary tables --------------------
def build_boundary_tables(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    *, bg_scope: str = "identified"
) -> Dict[str, pd.DataFrame]:
    """
    Build enrichment tables at peptide C- and N-terminal boundaries.
    Returns {'C': DataFrame, 'N': DataFrame}, each with columns:
      P1_count, P1p_count, bg_freq, P1_enrich, P1p_enrich, (optional p-values)
    """
    bg_freq = _bg_internal_freq(_bg_protein_subset(df_i, df_obs_map, bg_scope=bg_scope))
    seqs = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}

    def _collect(side: str) -> pd.DataFrame:
        rows = []
        for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
            seq = seqs.get(name, "")
            L = len(seq)
            if side == "C":
                # cut between e-1 | e
                if 0 < e <= L:
                    P1  = seq[e-1] if 0 <= e-1 < L else None
                    P1p = seq[e]   if e < L else None
            else:  # "N"
                # cut between s-1 | s
                if 0 <= s < L:
                    P1  = seq[s-1] if s-1 >= 0 else None
                    P1p = seq[s]   if s < L else None
            if P1 in AA20_SET and P1p in AA20_SET:
                rows.append((P1, P1p))
        if not rows:
            c1 = pd.Series(0, index=AA20, name="P1_count")
            c2 = pd.Series(0, index=AA20, name="P1p_count")
        else:
            dfb = pd.DataFrame(rows, columns=["P1","P1p"])
            c1 = dfb["P1"].value_counts().reindex(AA20, fill_value=0).rename("P1_count")
            c2 = dfb["P1p"].value_counts().reindex(AA20, fill_value=0).rename("P1p_count")
        out = pd.concat([c1, c2, bg_freq], axis=1)
        tot1  = max(1, int(c1.sum()))
        tot1p = max(1, int(c2.sum()))
        out["P1_enrich"]  = (out["P1_count"]  / tot1)  / out["bg_freq"]
        out["P1p_enrich"] = (out["P1p_count"] / tot1p) / out["bg_freq"]

        # Optional exact tests (approximate background table)
        if fisher_exact is not None and (tot1 > 0 or tot1p > 0):
            out["P1_pval"] = np.nan
            out["P1p_pval"] = np.nan
            # Build large background table from bg_freq (approximation)
            bg_total = int((_bg_internal_freq(_bg_protein_subset(df_i, df_obs_map, bg_scope))).sum() * 1e6) or 10**6
            for aa in AA20:
                bg_succ = max(1, int(round(bg_freq[aa] * bg_total)))
                bg_fail = max(1, bg_total - bg_succ)
                a = int(c1.get(aa, 0)); b = tot1 - a
                out.loc[aa, "P1_pval"]  = fisher_exact([[a, b],[bg_succ, bg_fail]], alternative="two-sided")[1]
                a2 = int(c2.get(aa, 0)); b2 = tot1p - a2
                out.loc[aa, "P1p_pval"] = fisher_exact([[a2, b2],[bg_succ, bg_fail]], alternative="two-sided")[1]
        return out.sort_index()

    return {"C": _collect("C"), "N": _collect("N")}

# -------------------- Derive C-terminal rule (look-behind) --------------------
def derive_c_rule(
    tables: Dict[str, pd.DataFrame],
    *, thr_strict=2.5, thr_balanced=1.8, thr_broad=1.2,
    alpha: Optional[float] = None,  # set to None to skip p-value filtering
    proline_guard: bool = False     # keep False unless you *want* (?!P)
) -> Dict[str, Dict[str, object]]:
    """
    Build residue sets from the C-terminus P1 enrichments and synthesize regex rules.
    Returns {'residues': {...}, 'rules': {...}}
    """
    enC = tables["C"]

    def _pick(series: pd.Series, pvals: Optional[pd.Series], thr: float, max_k=12) -> List[str]:
        s = series.dropna()
        if alpha is not None and pvals is not None and fisher_exact is not None:
            p = pvals.reindex(s.index)
            s = s[p.notna() & (p <= alpha)]
        s = s.sort_values(ascending=False)
        s2 = s[s >= thr]
        if s2.empty:
            s2 = s.head(5)
        return s2.index.tolist()[:max_k]

    res_strict   = _pick(enC["P1_enrich"], enC.get("P1_pval"), thr_strict)
    res_balanced = _pick(enC["P1_enrich"], enC.get("P1_pval"), thr_balanced)
    res_broad    = _pick(enC["P1_enrich"], enC.get("P1_pval"), thr_broad)

    def _rule(residues: List[str]) -> str:
        if not residues:
            r = r"(?<=.)"
        else:
            r = rf"(?<=[{''.join(residues)}])"
        if proline_guard:
            r += r"(?!P)"
        return r

    return {
        "residues": {
            "strict": res_strict,
            "balanced": res_balanced,
            "broad": res_broad,
        },
        "rules": {
            "strict":   _rule(res_strict),
            "balanced": _rule(res_balanced),
            "broad":    _rule(res_broad),
        }
    }

# -------------------- Scan N-terminal preferences (independent of C-rule) --------------------
def scan_nterm_preferences(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    *,
    S_P1: Iterable[str],            # C-rule residue set (P1 residues), e.g. {'L','F','E','W','D'}
    bg_scope: str = "identified",
    min_starts_with_prev: int = 25, # require at least this many N starts with an upstream residue present
    min_enrichment: float = 2.0,    # observed / background (conditional) threshold
    alpha: Optional[float] = 1e-3,  # Fisher one-sided (greater) p-value threshold; None=skip
    candidates: Optional[Iterable[str]] = None  # if None -> test all 20 AAs
) -> Dict[str, object]:
    """
    For each residue X, test if starts with X occur frequently even when the upstream P1 ∉ S_P1.
    Compares OBS: P(prev∉S | start=X, s>0) vs BG: P(prev∉S | curr=X) from internal bigrams.

    Returns dict with:
      - 'summary': DataFrame per residue with counts, proportions, enrichment, p-value
      - 'picked': list of residues passing thresholds
      - 'rule': look-ahead regex '(?=[...])' or None if none picked
    """
    S = set(x.upper() for x in S_P1)
    cand = list(candidates) if candidates else AA20

    # background bigrams from identified/all proteins
    df_bg = _bg_protein_subset(df_i, df_obs_map, bg_scope)
    n_curr = {aa: 0 for aa in AA20}  # total bigrams with curr=aa
    n_prev_notS_curr = {aa: 0 for aa in AA20}
    for _, r in df_bg.iterrows():
        seq = str(r["sequence"]).upper()
        for i in range(1, len(seq)):
            prev, curr = seq[i-1], seq[i]
            if prev in AA20_SET and curr in AA20_SET:
                n_curr[curr] += 1
                if prev not in S:
                    n_prev_notS_curr[curr] += 1

    # observed N-starts
    seqs = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}
    obs_counts = {aa: {"starts": 0, "starts_with_prev": 0, "prev_notS": 0} for aa in AA20}
    for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
        seq = seqs.get(name, "")
        L = len(seq)
        if not (0 <= s < L):
            continue
        X = seq[s]  # first residue of the peptide
        if X not in AA20_SET:
            continue
        obs_counts[X]["starts"] += 1
        if s > 0 and seq[s-1] in AA20_SET:
            obs_counts[X]["starts_with_prev"] += 1
            if seq[s-1] not in S:
                obs_counts[X]["prev_notS"] += 1

    rows = []
    picked = []
    for aa in cand:
        a_st = obs_counts[aa]["starts"]
        a_st_prev = obs_counts[aa]["starts_with_prev"]
        a_prev_notS = obs_counts[aa]["prev_notS"]

        # observed conditional (exclude s==0 where prev is undefined)
        p_obs = (a_prev_notS / a_st_prev) if a_st_prev > 0 else np.nan
        # background conditional given curr=aa
        p_bg = (n_prev_notS_curr[aa] / n_curr[aa]) if n_curr[aa] > 0 else np.nan
        enr = (p_obs / p_bg) if (np.isfinite(p_obs) and np.isfinite(p_bg) and p_bg > 0) else np.nan

        # Fisher exact on 2x2 conditioned on curr=aa
        pval = np.nan
        if fisher_exact is not None and a_st_prev > 0 and n_curr[aa] > 0:
            a = a_prev_notS
            b = a_st_prev - a
            c = n_prev_notS_curr[aa]
            d = n_curr[aa] - c
            try:
                pval = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
            except Exception:
                pval = np.nan

        rows.append({
            "aa": aa,
            "starts_total": a_st,
            "starts_with_prev": a_st_prev,
            "obs_prev_notS": a_prev_notS,
            "p_obs_prev_notS_given_start": p_obs,
            "bg_prev_notS_given_curr": p_bg,
            "enrichment": enr,
            "p_value": pval,
        })

        # pick?
        if (a_st_prev >= min_starts_with_prev and
            np.isfinite(enr) and enr >= min_enrichment and
            (alpha is None or (np.isfinite(pval) and pval <= alpha))):
            picked.append(aa)

    summary = (pd.DataFrame(rows)
                 .sort_values(["enrichment","starts_with_prev"], ascending=[False, False])
                 .reset_index(drop=True))

    rule = rf"(?=[{''.join(picked)}])" if picked else None
    return {"summary": summary, "picked": picked, "rule": rule}

# -------------------- Two-rule classification --------------------
def estimate_missed_cleavages_from_two_rules(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    rule_N: Optional[str],     # look-ahead, e.g. (?=[IM]); None -> ignore N-rule
    rule_C: Optional[str],     # look-behind, e.g. (?<=[LFEWD])(?!P)?; None -> ignore C-rule
) -> pd.DataFrame:
    """
    Classify boundaries with distinct N and C rules. A peptide is 'fully_enzymatic_2rule'
    if (start matches N-rule or is protein start) and (end matches C-rule or is protein end).
    Returns df with columns: name, peptide, start, end, N_is_cut, C_is_cut, fully_enzymatic_2rule.
    """
    pattN = re.compile(rule_N) if rule_N else None
    pattC = re.compile(rule_C) if rule_C else None
    seqs = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}

    out = []
    for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
        seq = seqs.get(name, "")
        L = len(seq)

        # compute cut indices for each rule
        cutsN = set(m.start() for m in pattN.finditer(seq)) if pattN else set()
        cutsC = set(m.end()   for m in pattC.finditer(seq)) if pattC else set()

        N_is = (s in cutsN) or (s == 0) if pattN else (s == 0)  # if no N-rule, only protein start is enzymatic
        C_is = (e in cutsC) or (e == L) if pattC else (e == L)  # if no C-rule, only protein end is enzymatic

        out.append({
            "name": name, "peptide": pep, "start": int(s), "end": int(e),
            "N_is_cut": bool(N_is), "C_is_cut": bool(C_is),
            "fully_enzymatic_2rule": int(bool(N_is and C_is)),
        })
    return pd.DataFrame(out)
# ---------------------------------------------------------------


# ---------- helpers you already have / variants ----------
def _clean_peptide_string(s: str) -> str:
    return re.sub(r'[^A-Z]', '', str(s).upper())

def map_observed_peptides_to_proteins(df_i: pd.DataFrame,
                                      obs_peptides: List[str],
                                      multi_map: str = "all") -> pd.DataFrame:
    """
    Return rows (name, peptide, start, end) for each occurrence of peptide in the protein.
    multi_map: 'all' → return all occurrences; 'first' → first occurrence per (name, peptide).
    """
    seq_map = {row["name"]: str(row["sequence"]).upper() for _, row in df_i.iterrows()}
    rows = []
    for pep in sorted(set(obs_peptides), key=len, reverse=True):
        for name, seq in seq_map.items():
            # find all occurrences, including overlaps
            for m in re.finditer(rf'(?={re.escape(pep)})', seq):
                s = m.start()
                rows.append((name, pep, s, s + len(pep)))
            if multi_map == "first" and rows and rows[-1][0] == name and rows[-1][1] == pep:
                # keep only the first for this (name, pep)
                break
    return pd.DataFrame(rows, columns=["name", "peptide", "start", "end"])



# ---------- 2) Missed-cleavage estimation given a rule ----------
def estimate_missed_cleavages_from_rule(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    rule_regex: str
) -> pd.DataFrame:
    """
    For each observed peptide mapping (name, peptide, start, end), count how many
    INTERNAL cleavage sites (matching rule_regex) lie strictly inside (start, end).
    Returns DataFrame with columns:
      name, peptide, start, end, missed_cleavages, fully_enzymatic (0/1)
    """
    patt = re.compile(rule_regex)
    rows = []
    for name, grp in df_obs_map.groupby("name"):
        seq = str(df_i.loc[df_i["name"] == name].iloc[0]["sequence"]).upper()
        # precompute cut positions for this protein
        cut_positions = [m.end() for m in patt.finditer(seq)]  # cut occurs between m.end()-1 | m.end()
        cut_positions = sorted(set(cut_positions))
        for _, r in grp.iterrows():
            s, e = int(r["start"]), int(r["end"])
            # fully enzymatic if s and e are cut boundaries
            full = int((s in cut_positions or s == 0) and (e in cut_positions or e == len(seq)))
            # internal sites strictly inside (s, e)
            internal = sum(1 for c in cut_positions if s < c < e)
            missed = max(internal - 1, 0)  # classic definition
            rows.append((name, r["peptide"], s, e, missed, full))
    return pd.DataFrame(rows, columns=["name","peptide","start","end","missed_cleavages","fully_enzymatic"])

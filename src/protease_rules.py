# protease_rules.py
# -----------------------------------------------------------------------------
# A compact module for learning protease rules from peptide evidence.
# - background restricted to identified proteins (configurable)
# - C-rule from P1 enrichment (look-behind)
# - optional N-rule from N-terminal independence (look-ahead)
# - missed-cleavage estimation (one-rule or two-rule)
# - re-fit enrichment using only fully-enzymatic peptides (optional)
# - journal-style plots + artifact saving
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False

try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None  # module works without SciPy (p-values -> NaN)

# ------------------------ constants / small utils ----------------------------

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_SET = set(AA20)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _aa_order_by(series: pd.Series, descending=True) -> List[str]:
    s = series.dropna().reindex(AA20).fillna(0)
    return s.sort_values(ascending=not descending).index.tolist()

def _bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini–Hochberg q-values; NaNs preserved."""
    p = pd.Series(p, dtype=float)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    m = p.notna().sum()
    if m == 0:
        return q
    s = p.sort_values()
    valid = s.dropna()
    ranks = np.arange(1, len(valid)+1)
    qq = valid.values * len(valid) / ranks
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    q.loc[valid.index] = np.clip(qq, 0, 1)
    return q

# ------------------------------ cleaning / mapping ---------------------------

def clean_peptide_string(s: str) -> str:
    """Keep capital letters only (strip mods/charges)."""
    return re.sub(r'[^A-Z]', '', str(s).upper())

def map_observed_peptides_to_proteins(
    df_i: pd.DataFrame,
    obs_peptides: List[str],
    multi_map: str = "all"
) -> pd.DataFrame:
    """
    Return (name, peptide, start, end) for each occurrence of peptide in proteins.
    multi_map: 'all' (default) or 'first' (first occurrence per (name, peptide)).
    """
    seq_map = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}
    rows = []
    for pep in sorted(set(obs_peptides), key=len, reverse=True):
        pat = re.compile(rf"(?={re.escape(pep)})")
        for name, seq in seq_map.items():
            found = False
            for m in pat.finditer(seq):
                s = m.start()
                rows.append((name, pep, s, s + len(pep)))
                found = True
                if multi_map == "first":
                    break
    return pd.DataFrame(rows, columns=["name", "peptide", "start", "end"])

# ------------------------------ background model -----------------------------

def _bg_protein_subset(df_i: pd.DataFrame, df_obs_map: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in {"identified", "all"}:
        raise ValueError("bg_scope must be 'identified' or 'all'")
    if scope == "identified":
        names = set(df_obs_map["name"].unique())
        return df_i[df_i["name"].isin(names)].copy()
    return df_i.copy()

def _bg_internal_freq(df_bg: pd.DataFrame) -> pd.Series:
    counts = pd.Series(0, index=AA20, dtype=int)
    total = 0
    for _, r in df_bg.iterrows():
        s = str(r["sequence"]).upper()
        if len(s) <= 2:
            continue
        for ch in s[1:-1]:
            if ch in AA20_SET:
                counts[ch] += 1
                total += 1
    if total == 0:
        raise ValueError("Background total is zero; check sequences/bg_scope.")
    return (counts / total).rename("bg_freq").replace(0, 1e-9)

# ------------------------------ boundary tables ------------------------------

def build_boundary_tables(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    *,
    bg_scope: str = "identified"
) -> Dict[str, pd.DataFrame]:
    """
    Build enrichment tables at peptide C- and N-terminal boundaries.
    Returns {'C': df, 'N': df} with columns:
      P1_count, P1p_count, bg_freq, P1_enrich, P1p_enrich, (optional P1_pval, P1p_pval)
    """
    bg_freq = _bg_internal_freq(_bg_protein_subset(df_i, df_obs_map, bg_scope))
    seqs = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}

    def _collect(side: str) -> pd.DataFrame:
        rows = []
        for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
            seq = seqs.get(name, "")
            L = len(seq)
            if side == "C":
                if 0 < e <= L:
                    P1  = seq[e-1] if 0 <= e-1 < L else None
                    P1p = seq[e]   if e < L else None
            else:
                if 0 <= s < L:
                    P1  = seq[s-1] if s-1 >= 0 else None
                    P1p = seq[s]   if s < L else None
            if P1 in AA20_SET and P1p in AA20_SET:
                rows.append((P1, P1p))
        if rows:
            dfb = pd.DataFrame(rows, columns=["P1","P1p"])
            c1 = dfb["P1"].value_counts().reindex(AA20, fill_value=0).rename("P1_count")
            c2 = dfb["P1p"].value_counts().reindex(AA20, fill_value=0).rename("P1p_count")
        else:
            c1 = pd.Series(0, index=AA20, name="P1_count")
            c2 = pd.Series(0, index=AA20, name="P1p_count")

        out = pd.concat([c1, c2, bg_freq], axis=1)
        tot1  = max(1, int(c1.sum()))
        tot1p = max(1, int(c2.sum()))
        out["P1_enrich"]  = (out["P1_count"]  / tot1)  / out["bg_freq"]
        out["P1p_enrich"] = (out["P1p_count"] / tot1p) / out["bg_freq"]

        # Optional exact tests via synthetic bg table
        if fisher_exact is not None and (tot1 > 0 or tot1p > 0):
            out["P1_pval"] = np.nan
            out["P1p_pval"] = np.nan
            bg_total = 10**6
            for aa in AA20:
                bg_succ = max(1, int(round(bg_freq[aa] * bg_total)))
                bg_fail = max(1, bg_total - bg_succ)
                a = int(c1.get(aa, 0)); b = tot1  - a
                a2= int(c2.get(aa, 0)); b2= tot1p - a2
                try:
                    out.loc[aa,"P1_pval"]  = fisher_exact([[a, b],[bg_succ,bg_fail]], alternative="two-sided")[1]
                    out.loc[aa,"P1p_pval"] = fisher_exact([[a2,b2],[bg_succ,bg_fail]], alternative="two-sided")[1]
                except Exception:
                    pass
        return out.sort_index()

    return {"C": _collect("C"), "N": _collect("N")}

# ------------------------------ C-rule selection -----------------------------

@dataclass
class RuleConfig:
    thr_strict: float = 2.5
    thr_balanced: float = 1.8
    thr_broad: float = 1.2
    min_count: int = 25
    use_pvals: bool = False        # use q-values gating
    fdr_alpha: float = 0.05        # BH threshold if use_pvals=True
    proline_guard: bool = False    # append (?!P)

def _pick_residues(enrich: pd.Series, counts: pd.Series,
                   pvals: Optional[pd.Series],
                   thr: float, min_count: int,
                   use_pvals: bool, fdr_alpha: float) -> List[str]:
    s = enrich.copy()
    s[counts < min_count] = np.nan
    if use_pvals and pvals is not None and pvals.notna().any():
        q = _bh_fdr(pvals)
        s[q > fdr_alpha] = np.nan
    s = s.dropna().sort_values(ascending=False)
    kept = s[s >= thr]
    if kept.empty:
        kept = s.head(5)  # fallback: top 5
    return kept.index.tolist()

def derive_c_rule(tables: Dict[str, pd.DataFrame], cfg: RuleConfig) -> Dict[str, Dict]:
    enC = tables["C"]
    pvals = enC["P1_pval"] if "P1_pval" in enC.columns else None

    res_strict   = _pick_residues(enC["P1_enrich"], enC["P1_count"], pvals,
                                  cfg.thr_strict, cfg.min_count, cfg.use_pvals, cfg.fdr_alpha)
    res_balanced = _pick_residues(enC["P1_enrich"], enC["P1_count"], pvals,
                                  cfg.thr_balanced, cfg.min_count, cfg.use_pvals, cfg.fdr_alpha)
    res_broad    = _pick_residues(enC["P1_enrich"], enC["P1_count"], pvals,
                                  cfg.thr_broad, cfg.min_count, cfg.use_pvals, cfg.fdr_alpha)

    def _rule(residues: List[str]) -> str:
        if not residues:
            r = r"(?<=.)"
        else:
            r = rf"(?<=[{''.join(residues)}])"
        if cfg.proline_guard:
            r += r"(?!P)"
        return r

    return {
        "residues": {
            "strict":   res_strict,
            "balanced": res_balanced,
            "broad":    res_broad,
        },
        "rules": {
            "strict":   _rule(res_strict),
            "balanced": _rule(res_balanced),
            "broad":    _rule(res_broad),
        }
    }

# ------------------------------ N-scan (independence) ------------------------

@dataclass
class NScanConfig:
    min_starts_with_prev: int = 25
    min_enrichment: float = 2.0
    use_pvals: bool = True
    fdr_alpha: float = 1e-3
    bg_scope: str = "identified"

def scan_nterm_preferences(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    S_P1: Iterable[str],
    cfg: NScanConfig
) -> Dict[str, object]:
    S = set(x.upper() for x in S_P1)
    df_bg = _bg_protein_subset(df_i, df_obs_map, cfg.bg_scope)

    # background P(prev∉S | curr=aa) from internal bigrams
    n_curr = {aa: 0 for aa in AA20}
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
    obs = {aa: {"starts": 0, "starts_with_prev": 0, "prev_notS": 0} for aa in AA20}
    for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
        seq = seqs.get(name, "")
        L = len(seq)
        if not (0 <= s < L):  # valid start
            continue
        X = seq[s]
        if X not in AA20_SET:
            continue
        obs[X]["starts"] += 1
        if s > 0 and seq[s-1] in AA20_SET:
            obs[X]["starts_with_prev"] += 1
            if seq[s-1] not in S:
                obs[X]["prev_notS"] += 1

    rows = []
    pvals = []
    for aa in AA20:
        st = obs[aa]["starts"]
        stp= obs[aa]["starts_with_prev"]
        a  = obs[aa]["prev_notS"]
        p_obs = (a / stp) if stp > 0 else np.nan
        p_bg  = (n_prev_notS_curr[aa] / n_curr[aa]) if n_curr[aa] > 0 else np.nan
        enr   = (p_obs / p_bg) if (np.isfinite(p_obs) and np.isfinite(p_bg) and p_bg > 0) else np.nan

        pval = np.nan
        if fisher_exact is not None and stp > 0 and n_curr[aa] > 0:
            b = stp - a
            c = n_prev_notS_curr[aa]
            d = n_curr[aa] - c
            try:
                pval = fisher_exact([[a,b],[c,d]], alternative="greater")[1]
            except Exception:
                pass
        pvals.append(pval)
        rows.append({
            "aa": aa,
            "starts_total": st,
            "starts_with_prev": stp,
            "obs_prev_notS": a,
            "p_obs_prev_notS_given_start": p_obs,
            "bg_prev_notS_given_curr": p_bg,
            "enrichment": enr,
            "p_value": pval
        })

    summary = pd.DataFrame(rows)
    summary["q_value"] = _bh_fdr(summary["p_value"]) if cfg.use_pvals else np.nan

    mask = (
        (summary["starts_with_prev"] >= cfg.min_starts_with_prev) &
        (summary["enrichment"] >= cfg.min_enrichment) &
        (~cfg.use_pvals | (summary["q_value"] <= cfg.fdr_alpha))
    )
    picked = summary.loc[mask, "aa"].tolist()
    rule = rf"(?=[{''.join(picked)}])" if picked else None

    return {"summary": summary.sort_values(["enrichment","starts_with_prev"], ascending=[False, False]).reset_index(drop=True),
            "picked": picked,
            "rule": rule}

# ------------------------------ missed cleavages ------------------------------

def estimate_missed_cleavages_from_rule(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    rule_regex: str
) -> pd.DataFrame:
    patt = re.compile(rule_regex)
    rows = []
    for name, grp in df_obs_map.groupby("name"):
        seq = str(df_i.loc[df_i["name"] == name].iloc[0]["sequence"]).upper()
        cuts = sorted({m.end() for m in patt.finditer(seq)})
        L = len(seq)
        for _, r in grp.iterrows():
            s, e = int(r["start"]), int(r["end"])
            full = int((s in cuts or s == 0) and (e in cuts or e == L))
            internal = sum(1 for c in cuts if s < c < e)
            missed = max(internal - 1, 0)
            rows.append((name, r["peptide"], s, e, missed, full))
    return pd.DataFrame(rows, columns=["name","peptide","start","end","missed_cleavages","fully_enzymatic"])

def estimate_missed_cleavages_from_two_rules(
    df_i: pd.DataFrame,
    df_obs_map: pd.DataFrame,
    rule_N: Optional[str],
    rule_C: Optional[str]
) -> pd.DataFrame:
    pattN = re.compile(rule_N) if rule_N else None
    pattC = re.compile(rule_C) if rule_C else None
    seqs = {r["name"]: str(r["sequence"]).upper() for _, r in df_i.iterrows()}
    out = []
    for name, pep, s, e in df_obs_map[["name","peptide","start","end"]].itertuples(index=False):
        seq = seqs.get(name, "")
        L = len(seq)
        cutsN = {m.start() for m in pattN.finditer(seq)} if pattN else set()
        cutsC = {m.end()   for m in pattC.finditer(seq)} if pattC else set()
        N_is = (s in cutsN) or (s == 0) if pattN else (s == 0)
        C_is = (e in cutsC) or (e == L) if pattC else (e == L)
        out.append({
            "name": name, "peptide": pep, "start": int(s), "end": int(e),
            "N_is_cut": bool(N_is), "C_is_cut": bool(C_is),
            "fully_enzymatic_2rule": int(bool(N_is and C_is)),
        })
    return pd.DataFrame(out)

# --------------------------------- plotting ----------------------------------

def plot_c_enrichment_lollipop(tables: Dict[str, pd.DataFrame],
                               picked_residues: Iterable[str],
                               title: str = "C-terminus P1 enrichment",
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
    enC = tables["C"].copy().reindex(AA20)
    order = _aa_order_by(enC["P1_enrich"])
    x = np.arange(len(order))
    y = enC.loc[order, "P1_enrich"].values
    counts = enC.loc[order, "P1_count"].values
    bg = enC.loc[order, "bg_freq"].values * 100

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.6), dpi=200)


    ax.hlines(1.0, -0.5, len(order)-0.5, colors="lightgray", linestyles="--", linewidth=0.8)
    for i, aa in enumerate(order):
        color = "crimson" if aa in set(picked_residues) else "steelblue"
        ax.vlines(i, 1.0, y[i], color=color, linewidth=2)
        ax.scatter(i, y[i], s=12 + 0.25*counts[i], color=color,
                   edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Enrichment at P1 (obs % / bg %)")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(x, bg, color="gray", alpha=0.15, width=0.6)
    ax2.set_ylim(0, (bg.max() if len(bg) else 1) * 1.1)
    ax2.set_yticks([])
    ax2.set_ylabel("bg %", color="gray")
    
    for s in ["right", "top"]:
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    return ax

def plot_n_scan(nscan: Dict[str,object],
                title: str = "N-scan: independence from C-rule") -> plt.Figure:
    df = nscan["summary"].copy()
    df = df[df["starts_with_prev"] > 0].copy()
    if _HAVE_SNS:
        # consistent AA→color mapping
        palette = dict(zip(df["aa"].unique(), sns.color_palette("tab20", n_colors=df["aa"].nunique())))

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), dpi=200,
                                 gridspec_kw={"width_ratios":[3,2]})

        for s in ["right", "top"]:
            for ax in axes:
                ax.spines[s].set_visible(False)

        # scatterplot
        sns.scatterplot(
            data=df, x="enrichment", y="p_obs_prev_notS_given_start",
            size="starts_with_prev", sizes=(20, 150),
            hue="aa", palette=palette, legend=False, ax=axes[0],
            edgecolor="black", linewidth=0.3
        )
        axes[0].axvline(1.0, color="gray", linestyle="--", lw=0.8)
        axes[0].set_xlabel("Enrichment vs bg (prev∉S | start=aa)")
        axes[0].set_ylabel("Observed P(prev∉S | start=aa)")
        axes[0].set_title(title, fontsize=9)

        picked = set(nscan["picked"])
        for _, r in df.iterrows():
            if r["aa"] in picked:
                axes[0].text(r["enrichment"]*1.01, r["p_obs_prev_notS_given_start"],
                             r["aa"], fontsize=7)

        # barplot with matching colors
        df2 = df.sort_values("starts_with_prev", ascending=False)
        sns.barplot(
            data=df2, x="aa", y="starts_with_prev",
            hue="aa",  # tell seaborn that 'aa' controls colors
            dodge=False,  # keep bars side-by-side instead of split
            palette=palette,
            ax=axes[1], ec='black', lw=0.5, legend=False
        )
        # highlight picked by black edge
        for i, r in df2.reset_index(drop=True).iterrows():
            if r["aa"] in picked:
                axes[1].patches[i].set_edgecolor("crimson")
                axes[1].patches[i].set_linewidth(1.5)

        axes[1].set_ylabel("# starts with upstream residue")
        axes[1].set_xlabel("Residue")
        axes[1].set_title("Evidence", fontsize=9)

        fig.tight_layout()
        return fig
    else:
        # fallback, monochrome
        fig, ax = plt.subplots(figsize=(5,3), dpi=200)
        for s in ["right", "top"]:
            ax.spines[s].set_visible(False)
        ax.scatter(df["enrichment"], df["p_obs_prev_notS_given_start"], s=20, c="k")
        ax.axvline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Enrichment")
        ax.set_ylabel("Observed P(prev∉S | start=aa)")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        return fig

def plot_boundary_compositions(tables: Dict[str,pd.DataFrame],
                               title: str = "P1 composition vs background") -> plt.Axes:
    enC = tables["C"].reindex(AA20); enN = tables["N"].reindex(AA20)
    df = pd.DataFrame({
        "aa": AA20,
        "C_obsP1%": enC["P1_count"] / max(enC["P1_count"].sum(), 1) * 100,
        "N_obsP1%": enN["P1_count"] / max(enN["P1_count"].sum(), 1) * 100,
        "bg%": enC["bg_freq"] * 100,
    }).melt(id_vars="aa", value_vars=["C_obsP1%","N_obsP1%","bg%"],
            var_name="series", value_name="percent")
    order = _aa_order_by(tables["C"]["P1_enrich"])
    if _HAVE_SNS:
        fig, ax = plt.subplots(1,1, figsize=(6.4, 2.8), dpi=200)

        for s in ["right", "top"]:
            ax.spines[s].set_visible(False)

        sns.barplot(data=df, x="aa", y="percent", hue="series", order=order, ec='black', lw=0.5, 
                    palette={"C_obsP1%":"#4C72B0","N_obsP1%":"#55A868","bg%":"#B0B0B0"}, ax=ax)
        ax.set_ylabel("% (P1 boundaries or background)")
        ax.set_xlabel("Residue at P1")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False)
        fig.tight_layout()
        return ax
    else:
        fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)

        for s in ["right", "top"]:
            ax.spines[s].set_visible(False)

        for key, color in zip(["C_obsP1%","N_obsP1%","bg%"], ["#4C72B0","#55A868","#B0B0B0"]):
            y = df[df["series"]==key].set_index("aa").reindex(order)["percent"]
            ax.plot(range(len(order)), y.values, marker="o", label=key, color=color)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
        ax.set_ylabel("%")
        ax.set_xlabel("Residue at P1")
        ax.set_title(title, fontsize=9)
        ax.legend()
        fig.tight_layout()
        return ax

# -------------------------------- orchestrator --------------------------------

@dataclass
class LearnerConfig:
    bg_scope: str = "identified"
    peptide_col: str = "Peptide"
    # defaults for rule selection
    rule_cfg: RuleConfig = RuleConfig()
    nscan_cfg: NScanConfig = NScanConfig()

class ProteaseRuleLearner:
    """Stateful pipeline around the helpers above."""
    def __init__(self, df_i: pd.DataFrame, df_exp: pd.DataFrame, cfg: LearnerConfig = LearnerConfig()):
        self.df_i = df_i.copy()
        self.df_exp = df_exp.copy()
        self.cfg = cfg

        # artifacts
        self.df_obs_map: Optional[pd.DataFrame] = None
        self.tables: Optional[Dict[str,pd.DataFrame]] = None
        self.c_model: Optional[Dict[str,Dict]] = None
        self.rule_C: Optional[str] = None
        self.S_P1: Optional[set] = None
        self.nscan: Optional[Dict[str,object]] = None
        self.rule_N: Optional[str] = None
        self.mc_df: Optional[pd.DataFrame] = None
        self.mc_percentile: Optional[float] = None
        self.mc_cutoff: Optional[float] = None

    # --------- builders ---------
    @classmethod
    def from_paths(cls, df_i_path: str, fragpipe_path: str, cfg: LearnerConfig = LearnerConfig()):
        df_i = pd.read_csv(df_i_path)
        df_exp = pd.read_csv(fragpipe_path, sep="\t")
        return cls(df_i, df_exp, cfg)

    def build_observations(self, multi_map: str = "all") -> "ProteaseRuleLearner":
        obs_peps = (self.df_exp[self.cfg.peptide_col].astype(str).map(clean_peptide_string).tolist())
        self.df_obs_map = map_observed_peptides_to_proteins(self.df_i, obs_peps, multi_map=multi_map)
        return self

    def build_boundary_tables(self) -> "ProteaseRuleLearner":
        assert self.df_obs_map is not None, "Call build_observations() first."
        self.tables = build_boundary_tables(self.df_i, self.df_obs_map, bg_scope=self.cfg.bg_scope)
        return self

    def derive_c_rule(self, rule_cfg: Optional[RuleConfig] = None) -> "ProteaseRuleLearner":
        assert self.tables is not None, "Call build_boundary_tables() first."
        rcfg = rule_cfg or self.cfg.rule_cfg
        self.c_model = derive_c_rule(self.tables, rcfg)
        self.rule_C = self.c_model["rules"]["broad"]
        self.S_P1 = set(self.c_model["residues"]["broad"])
        return self

    def run_nscan(self, ncfg: Optional[NScanConfig] = None) -> "ProteaseRuleLearner":
        assert self.S_P1 is not None, "Call derive_c_rule() first."
        cfg = ncfg or self.cfg.nscan_cfg
        self.nscan = scan_nterm_preferences(self.df_i, self.df_obs_map, self.S_P1, cfg)
        self.rule_N = self.nscan["rule"]
        return self

    # --------- missed cleavages / thresholds ---------
    def compute_missed_cleavages(self, rule_C: Optional[str] = None) -> "ProteaseRuleLearner":
        assert self.df_obs_map is not None, "Call build_observations() first."
        rC = rule_C or self.rule_C
        assert rC, "No C-rule defined."
        self.mc_df = estimate_missed_cleavages_from_rule(self.df_i, self.df_obs_map, rC)
        return self

    def set_missed_cleavage_percentile(self, percentile: float = 95.0, fully_specific: bool = True) -> "ProteaseRuleLearner":
        assert self.mc_df is not None, "Compute missed cleavages first."
        self.mc_percentile = float(percentile)
        if fully_specific:
            self.mc_cutoff = float(np.percentile(self.mc_df.query("fully_enzymatic == 1")["missed_cleavages"], percentile))
        else:
            self.mc_cutoff = float(np.percentile(self.mc_df["missed_cleavages"], percentile))
            
        return self

    def refit_on_fully_enzymatic(self,
                                 two_rule: bool = False,
                                 rule_cfg: Optional[RuleConfig] = None) -> "ProteaseRuleLearner":
        """Rebuild tables using only fully-enzymatic peptides (one- or two-rule) and re-derive C-rule."""
        assert self.df_obs_map is not None, "Call build_observations() first."
        if two_rule and self.rule_N and self.rule_C:
            mc2 = estimate_missed_cleavages_from_two_rules(self.df_i, self.df_obs_map, self.rule_N, self.rule_C)
            fe = set(mc2.loc[mc2["fully_enzymatic_2rule"] == 1, ["name","peptide","start","end"]]
                        .apply(tuple, axis=1))
        else:
            assert self.mc_df is not None, "Compute missed cleavages first (one-rule)."
            fe = set(self.mc_df.loc[self.mc_df["fully_enzymatic"] == 1, ["name","peptide","start","end"]]
                        .apply(tuple, axis=1))

        mask = self.df_obs_map.apply(lambda r: (r["name"], r["peptide"], r["start"], r["end"]) in fe, axis=1)
        df_obs_map_fe = self.df_obs_map.loc[mask].reset_index(drop=True)

        self.tables = build_boundary_tables(self.df_i, df_obs_map_fe, bg_scope=self.cfg.bg_scope)
        rcfg = rule_cfg or self.cfg.rule_cfg
        self.c_model = derive_c_rule(self.tables, rcfg)
        self.rule_C = self.c_model["rules"]["broad"]
        self.S_P1 = set(self.c_model["residues"]["broad"])
        return self

    # --------- export / plots ---------
    def save_results(self, out_dir: str, condition_label: str = "") -> Dict[str,str]:
        paths = {}
        _ensure_dir(out_dir)
        figs_dir = os.path.join(out_dir, "figs"); _ensure_dir(figs_dir)
        tabs_dir = os.path.join(out_dir, "tables"); _ensure_dir(tabs_dir)

        bundle = {
            "condition": condition_label,
            "bg_scope": self.cfg.bg_scope,
            "rule_C": self.rule_C,
            "rule_N": self.rule_N,
            "residues_C": self.c_model["residues"] if self.c_model else None,
            "rules_C": self.c_model["rules"] if self.c_model else None,
            "nscan_picked": (self.nscan or {}).get("picked", None),
            "mc_percentile": self.mc_percentile,
            "mc_cutoff": self.mc_cutoff,
        }
        with open(os.path.join(out_dir, "rules_bundle.json"), "w") as fh:
            json.dump(bundle, fh, indent=2)
        paths["rules_bundle_json"] = os.path.join(out_dir, "rules_bundle.json")

        if self.tables is not None:
            self.tables["C"].to_csv(os.path.join(tabs_dir, f"{condition_label}_boundary_C.csv"))
            self.tables["N"].to_csv(os.path.join(tabs_dir, f"{condition_label}_boundary_N.csv"))
            paths["boundary_C_csv"] = os.path.join(tabs_dir, f"{condition_label}_boundary_C.csv")
            paths["boundary_N_csv"] = os.path.join(tabs_dir, f"{condition_label}_boundary_N.csv")

        if self.nscan is not None:
            self.nscan["summary"].to_csv(os.path.join(tabs_dir, f"{condition_label}_Nscan_summary.csv"), index=False)
            paths["nscan_summary_csv"] = os.path.join(tabs_dir, f"{condition_label}_Nscan_summary.csv")

        if self.mc_df is not None:
            self.mc_df.to_csv(os.path.join(tabs_dir, f"{condition_label}_missed_cleavages.csv"), index=False)
            paths["missed_cleavages_csv"] = os.path.join(tabs_dir, f"{condition_label}_missed_cleavages.csv")

        # plots
        if self.tables is not None and self.S_P1 is not None:
            ax = plot_c_enrichment_lollipop(self.tables, self.S_P1,
                                            title=f"{condition_label}: C-terminus P1 enrichment")
            ax.figure.savefig(os.path.join(figs_dir, f"{condition_label}_C_P1_enrichment.png"), dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            paths["c_enrichment_png"] = os.path.join(figs_dir, f"{condition_label}_C_P1_enrichment.png")

        if self.nscan is not None:
            fig = plot_n_scan(self.nscan, title=f"{condition_label}: N-scan")
            fig.savefig(os.path.join(figs_dir, f"{condition_label}_N_scan.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths["n_scan_png"] = os.path.join(figs_dir, f"{condition_label}_N_scan.png")

        if self.tables is not None:
            ax2 = plot_boundary_compositions(self.tables, title=f"{condition_label}: P1 composition vs bg")
            ax2.figure.savefig(os.path.join(figs_dir, f"{condition_label}_P1_composition.png"), dpi=300, bbox_inches="tight")
            plt.close(ax2.figure)
            paths["p1_composition_png"] = os.path.join(figs_dir, f"{condition_label}_P1_composition.png")

        return paths

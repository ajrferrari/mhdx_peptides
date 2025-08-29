#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acid_peptide_pipeline.py

Digest proteins with configurable rules, build theoretical peptide sets,
annotate uniqueness (sequence-only by default), load FragPipe results,
map observed peptides to proteins, and generate summary plots + per-protein
sequence maps (Theoretical vs Observed) with aligned layout.

Usage (CLI):
  python acid_peptide_pipeline.py \
      --df-initial path/to/proteins.csv \
      --df-exp path/to/fragpipe/peptide.tsv \
      --outdir out_run \
      --enzyme Pepsin_broad --missed 4 --min-len 6 --max-len 60

Import (Notebook):
  from acid_peptide_pipeline import run_pipeline
  df_unique = run_pipeline(df_initial_csv, df_exp_tsv, outdir, enzyme="Pepsin_broad")
"""

from __future__ import annotations

import os
import re
import math
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pyteomics import parser, mass

# =============================================================================
# Cleavage rule registry (regex strings, look-behind; typical (?!P) proline guard)
# =============================================================================

RULES: Dict[str, str] = {
    "Pepsin_broad":          r'(?<=[FLIMVWYA])(?!P)',
    "Nep2_balanced":         r'(?<=[FLIMVWYAHKRNQD])(?!P)',
    "Pepsin_Nep2_union":     r'(?<=[FLIMVWYAHKRNQD])(?!P)',
    "Rhizopuspepsin_broad":  r'(?<=[FLIMVWYAHK])(?!P)',
    "Aspergillopepsin_XIII": r'(?<=[FLIMVWYAHK])(?!P)',
    "Trypsin":               r'(?<=[KR])(?!P)',
    "Chymotrypsin_strict":   r'(?<=[FWY])(?!P)',
    "Chymotrypsin_broad":    r'(?<=[FWYLM])(?!P)',
    # Non-specific: any boundary between two residues (still keeps proline guard for realism)
    "NonSpecific":           r'(?<=.)(?!P)',
}

def load_rules_bundle(path: str) -> dict:
    """Load a JSON produced by your rule-learning module."""
    with open(path, "r") as fh:
        return json.load(fh)

def resolve_c_rule_regex(bundle: dict,
                         *,
                         tier: str = "broad",
                         fallback_keys: tuple[str, ...] = ("rule_C",)) -> str:
    """
    Try a few common shapes:
      - bundle["rules"][tier]              (from derive_c_rule)
      - bundle[f"rule_{tier}"] or bundle["rule_C"]
      - bundle["c_model"]["rules"][tier]   (if you nested it)
    """
    # 1) Most common in our earlier design:
    try:
        r = bundle["rules"][tier]
        if isinstance(r, str) and r:
            return r
    except Exception:
        pass

    # 2) Explicit single-key fallbacks (e.g., "rule_C"):
    for k in fallback_keys + (f"rule_{tier}",):
        r = bundle.get(k)
        if isinstance(r, str) and r:
            return r

    # 3) Sometimes stored under c_model:
    try:
        r = bundle["c_model"]["rules"][tier]
        if isinstance(r, str) and r:
            return r
    except Exception:
        pass

    raise KeyError(
        f"Could not locate a C-rule regex in bundle. "
        f"Tried 'rules[{tier}]', {fallback_keys}, and 'c_model.rules[{tier}]'."
    )

# =============================================================================
# Digestion helpers
# =============================================================================

def peptides_enzymatic(
    seq: str,
    rule: str,
    missed: int = 4,
    min_len: int = 6,
    max_len: int = 60,
) -> List[str]:
    """Fully specific peptides using Pyteomics cleave (regex rule)."""
    return list(
        parser.cleave(
            seq, rule,
            missed_cleavages=missed,
            min_length=min_len,
            max_length=max_len,
            regex=True
        )
    )


def _cleavage_sites(seq: str, rule: str) -> List[int]:
    """Return sorted unique cleavage boundary indices (including 0 and len(seq))."""
    patt = re.compile(rule)
    sites = [0]
    sites += [m.end() for m in patt.finditer(seq)]
    sites.append(len(seq))
    return sorted(set(sites))


def peptides_semi_specific(
    seq: str,
    rule: str,
    missed: int = 4,
    min_len: int = 6,
    max_len: int = 60,
):
    """
    Semi-specific peptides: one enzymatic terminus, the other free.
    'missed' counts internal cleavage opportunities strictly inside.
    """
    sites = _cleavage_sites(seq, rule)
    seen = set()

    # Left enzymatic, right free
    for s in sites[:-1]:
        max_e = min(s + max_len, len(seq))
        for e in range(s + min_len, max_e + 1):
            internal = sum(1 for x in sites if s < x < e)
            if max(internal - 1, 0) <= missed:
                pep = seq[s:e]
                if pep not in seen:
                    seen.add(pep)
                    yield pep

    # Right enzymatic, left free
    for e in sites[1:]:
        min_s = max(0, e - max_len)
        for s in range(min_s, e - min_len + 1):
            if s in sites:  # fully specific already covered
                continue
            internal = sum(1 for x in sites if s < x < e)
            if max(internal - 1, 0) <= missed:
                pep = seq[s:e]
                if pep not in seen:
                    seen.add(pep)
                    yield pep


def _count_internal_sites(peptide: str, rule: str) -> int:
    """Count cleavage opportunities strictly inside a peptide (exclude termini)."""
    patt = re.compile(rule)
    return sum(1 for m in patt.finditer(peptide) if 0 < m.end() < len(peptide))


def digest_csv_to_peptide_buckets(
    df_i: pd.DataFrame,
    enzyme: str | None = None,
    max_missed: int = 4,
    *,
    rule_regex: str | None = None,   # <-- NEW
    semi_specific: bool = False,
    min_len: int = 6,
    max_len: int = 60,
    dedup: bool = True
) -> Dict[str, Dict[int, List[str]]]:
    """
    If rule_regex is provided, it OVERRIDES enzyme.
    Otherwise enzyme must be a known key in RULES.
    """
    if rule_regex:
        rule = rule_regex
    else:
        if not enzyme:
            raise KeyError("Provide either enzyme or rule_regex.")
        if enzyme not in RULES:
            raise KeyError(f"Unknown enzyme '{enzyme}'. Available: {list(RULES)}")
        rule = RULES[enzyme]

    out: Dict[str, Dict[int, List[str]]] = {}

    for _, row in df_i.iterrows():
        name = str(row['name']).strip()
        seq  = str(row['sequence']).strip().upper()
        if not seq:
            out[name] = {k: [] for k in range(max_missed + 1)}
            continue

        if semi_specific:
            peps = list(peptides_semi_specific(seq, rule, missed=max_missed,
                                               min_len=min_len, max_len=max_len))
        else:
            peps = list(peptides_enzymatic(seq, rule, missed=max_missed,
                                           min_len=min_len, max_len=max_len))
        peps = sorted(set(peps)) if dedup else sorted(peps)

        buckets: Dict[int, List[str]] = {k: [] for k in range(max_missed + 1)}
        for p in peps:
            mc = _count_internal_sites(p, rule)
            if mc <= max_missed:
                buckets[mc].append(p)
        for k in buckets:
            buckets[k].sort()
        out[name] = buckets

    return out


def dict_to_dataframe(
    res: Dict[str, Dict[int, List[str]]],
    *,
    column_missed: str = "n_missed_clavages"
) -> pd.DataFrame:
    """Flatten digest result into a DataFrame and compute monoisotopic mass."""
    rows: List[Tuple[str, str, int, float]] = []
    for name, buckets in res.items():
        for missed, peptides in buckets.items():
            for pep in peptides:
                try:
                    m = mass.calculate_mass(sequence=pep)  # neutral monoisotopic (includes +H2O)
                except Exception:
                    m = np.nan
                rows.append((name, pep, missed, m))
    return pd.DataFrame(rows, columns=["name", "peptide", column_missed, "monoisotopic_mass"])


# =============================================================================
# Uniqueness annotations
# =============================================================================

def annotate_uniqueness_by_sequence(
    df: pd.DataFrame,
    *,
    name_col: str = "name",
    peptide_col: str = "peptide",
    scope: str = "proteome",          # "proteome" or "protein"
    out_col: str = "is_unique",
    add_aux_cols: bool = False,
) -> pd.DataFrame:
    """
    Sequence-only uniqueness.

    scope="proteome": peptide is unique if it maps to exactly ONE distinct protein.
    scope="protein" : peptide is unique if it occurs once within that protein's row set.
    """
    if scope not in {"proteome", "protein"}:
        raise ValueError("scope must be 'proteome' or 'protein'.")

    df_out = df.copy()
    df_out[peptide_col] = df_out[peptide_col].astype(str).str.upper()

    if scope == "proteome":
        pairs = df_out[[name_col, peptide_col]].dropna().drop_duplicates()
        counts = (pairs.groupby(peptide_col)[name_col]
                        .nunique()
                        .rename("_n_proteins_with_pep")
                        .reset_index())
        df_out = df_out.merge(counts, on=peptide_col, how="left")
        df_out[out_col] = (df_out["_n_proteins_with_pep"] == 1).astype(int)
        if add_aux_cols:
            df_out["n_proteins_with_peptide"] = df_out["_n_proteins_with_pep"].fillna(0).astype(int)
        df_out.drop(columns=["_n_proteins_with_pep"], inplace=True, errors="ignore")
    else:
        counts = (df_out.groupby([name_col, peptide_col])
                        .size()
                        .rename("_n_in_protein")
                        .reset_index())
        df_out = df_out.merge(counts, on=[name_col, peptide_col], how="left")
        df_out[out_col] = (df_out["_n_in_protein"] == 1).astype(int)
        if add_aux_cols:
            df_out["n_occurrences_in_protein"] = df_out["_n_in_protein"].fillna(0).astype(int)
        df_out.drop(columns=["_n_in_protein"], inplace=True, errors="ignore")

    return df_out


def annotate_uniqueness_by_mass(
    df: pd.DataFrame,
    ppm_tol: float,
    *,
    mass_col: str = "monoisotopic_mass",
    peptide_col: str = "peptide",
    groupby: Optional[List[str]] = None,
    include_nearest: bool = False,
    nearest_col: str = "nearest_peptide",
    nearest_ppm_col: str = "nearest_ppm"
) -> pd.DataFrame:
    """
    Mass-based uniqueness within +/- ppm_tol. Optionally within groups (e.g. per protein).
    """
    def _process_block(block: pd.DataFrame) -> pd.DataFrame:
        block = block.copy()
        is_unique = np.ones(len(block), dtype=int)
        nearest = np.array([None] * len(block), dtype=object)
        nearest_ppm = np.full(len(block), np.nan, dtype=float)

        idx_valid = np.where(np.isfinite(block[mass_col].to_numpy()))[0]
        if idx_valid.size <= 1:
            block["is_unique"] = is_unique
            if include_nearest:
                block[nearest_col] = nearest
                block[nearest_ppm_col] = nearest_ppm
            return block

        m = block[mass_col].to_numpy()
        p = block[peptide_col].to_numpy()
        order = idx_valid[np.argsort(m[idx_valid])]
        m_sorted = m[order]

        for pos, i in enumerate(order):
            m1 = m_sorted[pos]
            delta = ppm_tol * 1e-6 * m1
            lo, hi = m1 - delta, m1 + delta

            best_ppm = np.inf
            best_pep = None

            # scan left
            k = pos - 1
            while k >= 0 and m_sorted[k] >= lo:
                is_unique[i] = 0
                ppm = abs(m_sorted[k] - m1) * 1e6 / m1
                if ppm < best_ppm:
                    best_ppm = ppm; best_pep = p[order[k]]
                k -= 1

            # scan right
            k = pos + 1
            while k < len(order) and m_sorted[k] <= hi:
                is_unique[i] = 0
                ppm = abs(m_sorted[k] - m1) * 1e6 / m1
                if ppm < best_ppm:
                    best_ppm = ppm; best_pep = p[order[k]]
                k += 1

            if include_nearest and np.isfinite(best_ppm):
                nearest[i] = best_pep
                nearest_ppm[i] = best_ppm

        block["is_unique"] = is_unique
        if include_nearest:
            block[nearest_col] = nearest
            block[nearest_ppm_col] = nearest_ppm
        return block

    if groupby:
        return (df.groupby(groupby, group_keys=False, dropna=False)
                  .apply(_process_block)
                  .reset_index(drop=True))
    else:
        return _process_block(df)


# =============================================================================
# FragPipe parsing + observed mapping
# =============================================================================

def _clean_peptide_string(s: str) -> str:
    """Uppercase A–Z only; strips mods/charges/punct."""
    return re.sub(r'[^A-Z]', '', str(s).upper())


def load_fragpipe_tsv(
    path: str,
    peptide_col_candidates=("Peptide", "peptide", "Sequence", "StrippedPeptide"),
    charge_col_candidates=("Charges", "Charge")
) -> pd.DataFrame:
    """Load FragPipe peptide table, standardize columns to ['Peptide','Charges','peptide_clean']."""
    df = pd.read_csv(path, sep='\t', low_memory=False)
    pep_col = next((c for c in peptide_col_candidates if c in df.columns), None)
    if pep_col is None:
        raise ValueError(f"Peptide column not found. Tried: {peptide_col_candidates}")
    df = df.rename(columns={pep_col: "Peptide"})
    if any(c in df.columns for c in charge_col_candidates):
        ch_col = next(c for c in charge_col_candidates if c in df.columns)
        df = df.rename(columns={ch_col: "Charges"})
    else:
        df["Charges"] = np.nan
    df["peptide_clean"] = df["Peptide"].map(_clean_peptide_string)
    df = df.loc[df["peptide_clean"].str.len() > 0].copy()
    return df


def map_observed_peptides_to_proteins(
    df_i: pd.DataFrame,
    obs_peptides: List[str]
) -> pd.DataFrame:
    """Return (name, peptide) rows where peptide is a substring of the protein sequence."""
    seq_map = {row["name"]: str(row["sequence"]).upper() for _, row in df_i.iterrows()}
    rows: List[Tuple[str, str]] = []
    for pep in sorted(set(obs_peptides), key=len, reverse=True):
        pat = re.compile(re.escape(pep))
        for name, seq in seq_map.items():
            if pat.search(seq):
                rows.append((name, pep))
    return pd.DataFrame(rows, columns=["name", "peptide"])


# =============================================================================
# Peptide-map plotting (aligned Theoretical vs Observed)
# =============================================================================

def _collect_peptides_for_name(df: pd.DataFrame, name: str) -> List[str]:
    """Collect unique peptides for a protein name, sorted long→short."""
    s = df.loc[df["name"] == name, "peptide"]
    return (s.dropna().astype(str).str.upper()
             .drop_duplicates()
             .sort_values(key=lambda x: x.str.len(), ascending=False).tolist())


def _find_occurrences(seq: str, pep: str) -> List[int]:
    """All start offsets for 'pep' in 'seq' (allow overlaps)."""
    pat = re.compile(rf"(?={re.escape(pep)})")
    return [m.start() for m in pat.finditer(seq)]


def _segments_for(
    df_i: pd.DataFrame, name: str, peps: List[str], residues_per_line: int
):
    """Compute per-line segments, packed into non-overlapping tracks."""
    seq = str(df_i.loc[df_i["name"] == name].iloc[0]["sequence"]).strip().upper()
    L = len(seq)
    n_lines = math.ceil(L / residues_per_line)
    line_ranges = [(i * residues_per_line, min((i + 1) * residues_per_line, L))
                   for i in range(n_lines)]

    segs_per_line = {i: [] for i in range(n_lines)}
    for pep in peps:
        for s0 in _find_occurrences(seq, pep):
            e0 = s0 + len(pep)
            for li, (ls, le) in enumerate(line_ranges):
                ss, ee = max(s0, ls), min(e0, le)
                if ss < ee:
                    segs_per_line[li].append({
                        "peptide": pep, "start": s0, "end": e0,
                        "line_idx": li, "col_start": ss - ls, "col_end": ee - ls
                    })

    tracks_per_line: Dict[int, int] = {}
    for li in range(n_lines):
        segs = segs_per_line[li]
        segs.sort(key=lambda d: (d["col_start"], -(d["col_end"] - d["col_start"])))
        tracks: List[int] = []
        for s in segs:
            for t_idx, last_end in enumerate(tracks):
                if s["col_start"] >= last_end:
                    tracks[t_idx] = s["col_end"]; break
            else:
                tracks.append(s["col_end"])
        tracks_per_line[li] = len(tracks)
    return line_ranges, tracks_per_line, segs_per_line, seq, L


def build_peptide_colormap(peptides: List[str], palette: str = "tab20") -> Dict[str, tuple]:
    """Consistent colors for a set of peptides."""
    peps = pd.Series(peptides, dtype=str).str.upper().dropna().unique().tolist()
    peps = sorted(peps, key=len, reverse=True)
    N = max(10, len(peps))
    cmap = mpl.colormaps.get_cmap(palette).resampled(N)
    colors = cmap(np.linspace(0, 1, N, endpoint=False))
    return {pep: colors[i % N] for i, pep in enumerate(peps)}


def estimate_panel_height(
    df_i: pd.DataFrame,
    name: str,
    *,
    layout_df: Optional[pd.DataFrame],
    residues_per_line: int = 30,
    dpi: int = 300,
    # must match plot_unique_peptide_map defaults/params:
    line_top_margin: float = 0.05,
    interline_gap: float = 0.03,
    final_bottom_margin: float = 0.05,
    text_height: float = 0.30,
    rect_h: float = 0.08,
    rect_gap: float = -0.15,
    track_gap: float = 0.035,
    # auto-height controls:
    min_track_px: int = 12,
    min_text_px: int = 10,
    height_scale: float = 0.45,
    min_inches: float = 1.3,
    max_inches: Optional[float] = None,
) -> float:
    """
    Compute required figure height (inches) for a single panel so that
    rectangle tracks and residue text have at least the requested pixel height.
    Use the union of peptides in `layout_df` to mirror the plot layout.
    """
    # --- gather peptides used for layout for this name
    def _collect(df, nm):
        if df is None or df.empty:
            return []
        s = df.loc[df["name"] == nm, "peptide"]
        return (s.dropna().astype(str).str.upper()
                 .drop_duplicates()
                 .sort_values(key=lambda x: x.str.len(), ascending=False)
                 .tolist())

    peps_layout = _collect(layout_df, name)

    # --- sequence + line segmentation
    seq = str(df_i.loc[df_i["name"] == name].iloc[0]["sequence"]).strip().upper()
    L = len(seq)
    n_lines = math.ceil(L / residues_per_line)
    line_ranges = [(i * residues_per_line, min((i + 1) * residues_per_line, L))
                   for i in range(n_lines)]

    # --- segments only to count needed tracks per line
    _, tracks_per_line_layout, _, _, _ = _segments_for(df_i, name, peps_layout, residues_per_line)

    # --- vertical budget per line (match plot_unique_peptide_map)
    block_heights = []
    for li in range(n_lines):
        n_tracks = tracks_per_line_layout.get(li, 0)
        pad_before = line_top_margin if li == 0 else interline_gap
        block_h = pad_before + text_height
        if n_tracks > 0:
            block_h += rect_gap + n_tracks * (rect_h + track_gap)
        block_heights.append(block_h)

    total_height = sum(block_heights) + final_bottom_margin
    if total_height <= 0:
        return min_inches

    # --- dpi-aware inches so tracks / text have readable pixels
    eps = 1e-9
    need_for_tracks = (min_track_px * total_height) / max(rect_h, eps) / dpi
    need_for_text   = (min_text_px   * total_height) / max(text_height, eps) / dpi
    need_by_scale   = total_height * height_scale  # already in "layout units"; scale is unitless
    # Convert need_by_scale to inches by interpreting scale as inches per unit:
    # prior code used: fig_h = total_height * height_scale
    fig_h = max(min_inches, need_for_tracks, need_for_text, need_by_scale)
    if max_inches is not None:
        fig_h = min(fig_h, max_inches)
    return fig_h


def plot_unique_peptide_map(
    df_i: pd.DataFrame,
    df_peps: pd.DataFrame,
    name: str,
    ax=None,
    *,
    residues_per_line: int = 30,
    font_size: int = 6,
    dpi: int = 600,
    palette: str = "tab20",
    line_top_margin: float = 0.05,
    interline_gap: float = 0.03,
    final_bottom_margin: float = 0.05,
    text_height: float = 0.30,
    rect_h: float = 0.08,
    rect_gap: float = -0.15,
    track_gap: float = 0.035,
    edge_lw: float = 0.3,
    rect_left_pad: float = 0.04,
    rect_right_pad: float = 0.05,
    min_rect_width: float = 0.10,
    color_map: Optional[Dict[str, tuple]] = None,
    layout_df: Optional[pd.DataFrame] = None,
    label_header: Optional[str] = None,
    auto_height: bool = True,
    height_scale: float = 0.45,   # used if auto_height=False or as a floor
    min_track_px: int = 12,       # min pixel height per track
    min_text_px: int = 10,        # min pixel height for the AA text block
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw sequence text and rectangles for peptides in df_peps for a given protein name.
    If layout_df is provided, vertical spacing is normalized using the union of peptides
    so Theoretical and Observed panels align.
    """
    rows = df_i.loc[df_i["name"] == name]
    if rows.empty:
        raise ValueError(f"No sequence found for name='{name}'.")
    seq = str(rows.iloc[0]["sequence"]).strip().upper()
    L = len(seq)

    peps_draw = _collect_peptides_for_name(df_peps, name)
    peps_layout = _collect_peptides_for_name(layout_df, name) if layout_df is not None else peps_draw

    line_ranges, tracks_per_line_layout, _, _, _ = _segments_for(df_i, name, peps_layout, residues_per_line)
    _, _, segs_per_line_draw, _, _ = _segments_for(df_i, name, peps_draw, residues_per_line)

    if color_map is None:
        color_map = build_peptide_colormap(peps_layout if peps_layout else peps_draw, palette=palette)

    # vertical layout
    block_heights: List[float] = []
    n_lines = len(line_ranges)
    for li in range(n_lines):
        n_tracks = tracks_per_line_layout.get(li, 0)
        pad_before = line_top_margin if li == 0 else interline_gap
        block_h = pad_before + text_height
        if n_tracks > 0:
            block_h += rect_gap + n_tracks * (rect_h + track_gap)
        block_heights.append(block_h)

    y_tops = np.cumsum([0.0] + block_heights[:-1])
    total_height = sum(block_heights) + final_bottom_margin

    created = False
    if ax is None:
        # --- NEW: dpi-aware auto height ---
        if auto_height:
            # inches required so that a single track rectangle has at least min_track_px
            # rect_h and text_height are in the same "layout units" that sum to total_height
            # pixels_per_unit = fig_h_in * dpi / total_height
            # require: rect_h * pixels_per_unit >= min_track_px
            # => fig_h_in >= min_track_px * total_height / (rect_h * dpi)
            eps = 1e-9
            need_for_tracks = (min_track_px * total_height) / max(rect_h, eps) / dpi

            # also keep letters readable: text block per line gets at least min_text_px
            need_for_text = (min_text_px * total_height) / max(text_height, eps) / dpi

            # baseline scale (your previous behavior) as a soft floor
            need_by_scale = total_height * height_scale / dpi  # convert px->in by dividing dpi
            
            fig_h = max(1.3, need_for_tracks, need_for_text, need_by_scale)

        else:
            # legacy behavior
            fig_h = max(1.3, total_height * height_scale)

        fig_w = max(3.0, residues_per_line * 0.14)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        created = True
    else:
        fig = ax.figure

    ax.set_xlim(0, residues_per_line)
    ax.set_ylim(0, total_height)
    ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_facecolor("white")
    for s in ["left", "right", "top", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    if created:
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04)

    ax.text(0.0, 1.05, f"{name}  L={L}, N={len(peps_draw)}",
            ha="left", va="top", fontweight='bold', fontsize=font_size, transform=ax.transAxes)
    if label_header:
        ax.text(1.0, 1.05, label_header, ha="right", va="top", fontsize=font_size+1, transform=ax.transAxes)

    for li, (ls, le) in enumerate(line_ranges):
        line_seq = seq[ls:le]
        y_text_top = y_tops[li] + (line_top_margin if li == 0 else interline_gap)

        # sequence
        for col, aa in enumerate(line_seq):
            ax.text(col + 0.02, y_text_top, aa,
                    ha="left", va="top", family="DejaVu Sans Mono", fontsize=font_size)

        # rectangles
        line_segs = segs_per_line_draw.get(li, [])
        if line_segs:
            line_segs_sorted = sorted(line_segs, key=lambda d: (d["col_start"], -(d["col_end"] - d["col_start"])))
            last_ends: List[int] = []
            for s in line_segs_sorted:
                for t_idx, last_end in enumerate(last_ends):
                    if s["col_start"] >= last_end:
                        s["track"] = t_idx; last_ends[t_idx] = s["col_end"]; break
                else:
                    s["track"] = len(last_ends); last_ends.append(s["col_end"])

            for s in line_segs_sorted:
                track = s["track"]
                xL_raw, xR_raw = float(s["col_start"]), float(s["col_end"])
                x0 = xL_raw + rect_left_pad
                x1 = xR_raw - rect_right_pad
                if x1 - x0 < min_rect_width:
                    x1 = x0 + min_rect_width
                w = x1 - x0
                y0 = (y_text_top + text_height +
                      (rect_gap if tracks_per_line_layout.get(li, 0) > 0 else 0.0) +
                      track * (rect_h + track_gap))
                rect = Rectangle(
                    (x0, y0), w, rect_h,
                    facecolor=color_map.get(s["peptide"], (0.6, 0.6, 0.6, 0.85)),
                    edgecolor="black", linewidth=edge_lw, alpha=0.85
                )
                ax.add_patch(rect)

    return fig, ax


# =============================================================================
# Summary plots
# =============================================================================

def plot_unique_nonunique_all(
    df_unique: pd.DataFrame,
    out_png: str,
    font_size: int = 7
) -> None:
    """
    Single stacked horizontal bar plot: unique vs non-unique counts per protein,
    sorted by # unique (desc).
    """
    g = (df_unique.groupby(["name", "is_unique"])["peptide"]
         .nunique()
         .reset_index(name="n"))
    names = sorted(df_unique["name"].unique().tolist())
    pivot = g.pivot(index="name", columns="is_unique", values="n").reindex(names).fillna(0)
    pivot = pivot.rename(columns={0: "non_unique", 1: "unique"})
    pivot = pivot.sort_values("unique", ascending=False)

    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(pivot))), dpi=300)
    ax.barh(y, pivot["non_unique"], color="lightcoral", edgecolor="black", linewidth=0.2, label="Non-unique")
    ax.barh(y, pivot["unique"], left=pivot["non_unique"], color="cornflowerblue", edgecolor="black", linewidth=0.2, label="Unique")
    ax.set_yticks(y, pivot.index.tolist(), fontsize=font_size)
    ax.set_xlabel("Peptide count", fontsize=font_size)
    ax.legend(fontsize=font_size, frameon=False, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", ls="--", lw=0.3, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(
    df_initial_csv: str,
    df_exp_tsv: str,
    outdir: str,
    *,
    enzyme: str = "Pepsin_broad",
    # NEW: external rules bundle
    rules_bundle_path: str | None = None,
    rules_bundle_tier: str = "broad",    # "strict" | "balanced" | "broad"
    missed_cleavages: int = 4,
    semi_specific: bool = False,
    min_len: int = 6,
    max_len: int = 60,
    ppm_tol: float = 10.0,
    residues_per_line: int = 30,
    palette: str = "tab20",
    exclude_charge_one: bool = True,
    top_n_maps: Optional[int] = None,
) -> pd.DataFrame:
    ...
    # 1) Load initial sequences
    df_i = pd.read_csv(df_initial_csv)
    ...

    # --- NEW: pick rule source ---
    custom_regex: str | None = None
    if rules_bundle_path:
        bundle = load_rules_bundle(rules_bundle_path)
        custom_regex = resolve_c_rule_regex(bundle, tier=rules_bundle_tier)

    # 2) Digest → df_unique
    res = digest_csv_to_peptide_buckets(
        df_i,
        enzyme=None if custom_regex else enzyme,
        rule_regex=custom_regex,                 # <-- NEW
        max_missed=missed_cleavages,
        semi_specific=semi_specific,
        min_len=min_len,
        max_len=max_len,
        dedup=True,
    )

    df_unique = dict_to_dataframe(res)
    # Default “uniqueness” = sequence-only across proteome
    df_unique = annotate_uniqueness_by_sequence(df_unique, scope="proteome")

    # 3) Load FragPipe TSV and prepare experimental peptides
    df_exp = load_fragpipe_tsv(df_exp_tsv)
    if exclude_charge_one and "Charges" in df_exp.columns:
        with np.errstate(all="ignore"):
            df_exp = df_exp.loc[df_exp["Charges"].astype(str) != "1"].copy()
    exp_peptides = df_exp["peptide_clean"].astype(str).tolist()

    # 4) Map observed peptides to proteins by containment
    df_obs_map = map_observed_peptides_to_proteins(df_i, exp_peptides)  # ['name','peptide']

    # 5) Mark observed in df_unique (string match within the same protein)
    key = df_unique[["name", "peptide"]].copy()
    key["peptide"] = key["peptide"].str.upper()
    obs_key = df_obs_map.copy()
    obs_key["peptide"] = obs_key["peptide"].str.upper()
    key["observed"] = key.merge(obs_key.assign(observed=1), how="left",
                                on=["name", "peptide"])["observed"].fillna(0).astype(int)
    df_unique = df_unique.merge(key[["name", "peptide", "observed"]],
                                on=["name", "peptide"], how="left")

    # 6) Make folders
    df_dir  = os.path.join(outdir, "dataframe")
    fig_dir = os.path.join(outdir, "figs")
    os.makedirs(df_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 7) Save df_unique JSON
    out_json = os.path.join(df_dir, "df_unique.json")
    with open(out_json, "w") as fh:
        json.dump(json.loads(df_unique.to_json(orient="records")), fh, indent=2)

    # 8) Save “all proteins in one plot”: stacked unique vs non-unique
    out_bars = os.path.join(fig_dir, "unique_vs_nonunique_by_protein.png")
    plot_unique_nonunique_all(df_unique, out_bars)

    # 9) Per-protein peptide maps (Theoretical vs Observed) with aligned layout
    # order proteins by observed unique count (desc)
    obs_counts = (df_unique.loc[df_unique["observed"] == 1]
                  .groupby("name")["peptide"].nunique()
                  .reindex(df_i["name"].tolist(), fill_value=0)
                  .sort_values(ascending=False))
    names = obs_counts.index.tolist()
    if top_n_maps is not None:
        names = names[:top_n_maps]

    for name in names:
        # theoretical unique for this protein
        df_theo_i = df_unique.query("name == @name and is_unique == 1")[["name", "peptide"]].copy()
        # observed (mapped) for this protein
        df_obs_i = df_obs_map.query("name == @name")[["name", "peptide"]].copy()

        # shared layout + colors (union ensures identical panel heights)
        layout_df = pd.concat([df_theo_i, df_obs_i], ignore_index=True)
        peps_union = pd.concat([df_theo_i["peptide"], df_obs_i["peptide"]], ignore_index=True)
        cmap_shared = build_peptide_colormap(peps_union, palette=palette)

        # shared layout + colors (union) — as you already do:
        layout_df = pd.concat([df_theo_i, df_obs_i], ignore_index=True)
        
        # compute a height that fits ALL rectangles for this protein (both panels share layout)
        fig_h = estimate_panel_height(
            df_i, name,
            layout_df=layout_df,
            residues_per_line=residues_per_line,
            dpi=300,                # match your call below
            rect_h=0.06,            # match the rect_h you pass to plot_unique_peptide_map
            line_top_margin=0.05,
            interline_gap=0.00,
            rect_gap=-0.15,
            track_gap=0.035,
            min_track_px=12,        # bump these if still cramped
            min_text_px=10,
            height_scale=0.45,      # acts as a soft floor
            min_inches=1.8          # a slightly higher floor is often nicer
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(7, fig_h), dpi=300)

        plot_unique_peptide_map(
            df_i, df_theo_i, name=name, ax=axes[0],
            residues_per_line=residues_per_line, rect_h=0.06,
            interline_gap=0.0, rect_gap=-0.15, rect_right_pad=0.5,
            color_map=cmap_shared, layout_df=layout_df, label_header="Theoretical"
        )
        plot_unique_peptide_map(
            df_i, df_obs_i if not df_obs_i.empty else df_obs_i.assign(peptide=[]),
            name=name, ax=axes[1],
            residues_per_line=residues_per_line, rect_h=0.06,
            interline_gap=0.0, rect_gap=-0.15, rect_right_pad=0.5,
            color_map=cmap_shared, layout_df=layout_df, label_header="Observed"
        )

        # far-right labels (redundant with label_header, but explicit)
        axes[0].text(1, 1.05, "Theoretical", transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=7)
        axes[1].text(1, 1.05, "Observed",   transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=7)

        plt.tight_layout(w_pad=0.6)
        out_png = os.path.join(fig_dir, f"{re.sub(r'[^A-Za-z0-9._-]','_',name)}.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

    return df_unique


# =============================================================================
# CLI
# =============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Acid protease proteomics: theoretical digestion, uniqueness, mapping, and plots."
    )
    p.add_argument("--df-initial", required=True, help="CSV with columns: name,sequence[,mono_mass].")
    p.add_argument("--df-exp", required=True, help="FragPipe TSV (tab-separated).")
    p.add_argument("--outdir", required=True, help="Output folder (creates dataframe/ and figs/).")
    p.add_argument("--enzyme", default="Pepsin_broad", choices=list(RULES.keys()))
    p.add_argument("--missed", type=int, default=4, help="Max missed cleavages.")
    p.add_argument("--semi-specific", action="store_true", help="Use semi-specific digestion.")
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--ppm", type=float, default=10.0, help="(kept for compat; not used in default uniqueness)")
    p.add_argument("--residues-per-line", type=int, default=30)
    p.add_argument("--palette", default="tab20")
    p.add_argument("--include-charge1", action="store_true",
                   help="Keep charge 1 (by default excluded if 'Charges' present).")
    p.add_argument("--top-n-maps", type=int, default=None,
                   help="Limit number of peptide-map PNGs (desc by observed unique).")
    p.add_argument("--rules-bundle", default=None,
                   help="Path to rules_bundle.json with learned C-rule.")
    p.add_argument("--rules-bundle-tier", default="broad",
                   choices=["strict", "balanced", "broad"],
                   help="Which tier of C-rule to use from the bundle.")
    return p


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    run_pipeline(
        df_initial_csv=args.df_initial,
        df_exp_tsv=args.df_exp,
        outdir=args.outdir,
        enzyme=args.enzyme,
        rules_bundle_path=args.rules_bundle,         
        rules_bundle_tier=args.rules_bundle_tier,    
        missed_cleavages=args.missed,
        semi_specific=args.semi_specific,
        min_len=args.min_len,
        max_len=args.max_len,
        ppm_tol=args.ppm,
        residues_per_line=args.residues_per_line,
        palette=args.palette,
        exclude_charge_one=not args.include_charge1,
        top_n_maps=args.top_n_maps,
    )


if __name__ == "__main__":
    main()

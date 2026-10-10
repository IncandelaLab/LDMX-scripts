#!/usr/bin/env python3
"""
plot_roc_curves.py
==================
Plot ROC curves comparing 2 (or 1) BDTs.
Produces both log-x and linear x-axis plots.
AUC values are shown ONLY on the linear plot.

Usage:
    python plot_roc_curves.py
    (Edit the CONFIGURATION section below before running.)
"""

import os
import sys
import glob
import numpy as np
import awkward as ak
import uproot
from array import array
from sklearn import metrics
import ROOT as r

sys.setrecursionlimit(10000)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────

# --- BDTs to compare ---
# To compare 2 BDTs, put both entries. To plot only 1 BDT, comment out one.

BDT_BRANCHES = {
    "BDT w Truth":       "EcalVeto_SegMip_Truth_BDT",      # 1
    "BDT w Recoil":   "EcalVeto_SegMip_Digi_BDT",       # 2
    # "SP w SP":           "EcalVeto_SP_Legacy_Truth_BDT",    # 3
    # "SP w Recoil":       "EcalVeto_SP_Legacy_Digi_BDT",     # 4
}

# --- Pairs to compare (1-indexed into BDT_BRANCHES order) ---
# e.g. (1,2) compares the 1st and 2nd entries above
BDT_PAIRS = [
    (1, 2),
    # (2, 3),
    # (2, 4),
    # (2, 5),
]

# Input directories
BASE_SAMPLES = "/home/jihoonyoo/LDMX/8GeV_Samples/testing"
ECAL_PN_DIR  = os.path.join(BASE_SAMPLES, "ecal_pn_v15_8gev_final_BDT")
SIGNAL_DIR   = os.path.join(BASE_SAMPLES, "signal_final_BDT")


# --- Signal mass points ---
SIGNAL_MASSES = {
    "1 MeV":    "1MeV",
    "10 MeV":   "10MeV",
    "100 MeV":  "100MeV",
    "1000 MeV": "1000MeV",
}

# --- Output directory ---
OUTPUT_BASE = "/home/jihoonyoo/LDMX/v4.5.11/ldmx-sw/bdt_analysis/BDT_Comparisons"

# --- File limits ---
MAX_FILES_BKG     = 500
MAX_FILES_SIG     = 40
TAKE_LAST_BKG     = True

# --- Trigger branch for signal ---
TRIGGER_BRANCH = "Trigger_sim"


# ─────────────────────────────────────────────────────────────
#  SampleContainer
# ─────────────────────────────────────────────────────────────

class SampleContainer:
    """Loads .root files and extracts BDT disc values."""
    def __init__(self, path, is_signal, ecalveto_branch,
                 trigger_branch=None, max_files=None, take_last=False):
        self.path = path
        self.is_signal = is_signal
        self.ecalveto_branch = ecalveto_branch
        self.trigger_branch = trigger_branch
        self.max_files = max_files
        self.take_last = take_last
        self.files = self._collect_files()
        print(f"[SampleContainer] {len(self.files)} files from: {path}")

    def _collect_files(self):
        files = []
        if os.path.isfile(self.path) and self.path.endswith('.root'):
            files = [self.path]
        elif os.path.isdir(self.path):
            for dirpath, _, filenames in os.walk(self.path):
                for f in filenames:
                    if f.endswith('.root'):
                        files.append(os.path.join(dirpath, f))
            files.sort()
        else:
            files = sorted(glob.glob(self.path))
        if not files:
            print(f"[WARNING] No .root files found at: {self.path}")
        if self.max_files is not None:
            if self.take_last:
                files = files[-self.max_files:]
            else:
                files = files[:self.max_files]
        return files

    def get_disc_values(self):
        disc_branch = f"{self.ecalveto_branch}/disc_value_"
        branches_to_load = [disc_branch]
        if self.is_signal and self.trigger_branch:
            trig_branch = f"{self.trigger_branch}/pass_"
            branches_to_load.append(trig_branch)

        all_disc = []
        for fpath in self.files:
            try:
                with uproot.open(fpath) as f:
                    tree = f["LDMX_Events"]
                    arrays = tree.arrays(branches_to_load, library="ak")
                    disc = arrays[disc_branch]
                    if self.is_signal and self.trigger_branch:
                        trig_pass = arrays[trig_branch]
                        disc = disc[trig_pass]
                    all_disc.append(ak.to_numpy(ak.flatten(disc, axis=None)))
            except Exception as e:
                print(f"[WARNING] Skipping {fpath}: {e}")
                continue
        if all_disc:
            return np.concatenate(all_disc)
        return np.array([])


# ─────────────────────────────────────────────────────────────
#  PyROOT Plotting Helpers
# ─────────────────────────────────────────────────────────────

def _draw_ldmx_header():
    tex_ldmx = r.TLatex(0.15, 0.92, "LDMX")
    tex_ldmx.SetNDC(); tex_ldmx.SetTextFont(61)
    tex_ldmx.SetTextSize(0.0675); tex_ldmx.SetLineWidth(2)
    tex_wip = r.TLatex(0.33, 0.92, "Work in Progress")
    tex_wip.SetNDC(); tex_wip.SetTextFont(52)
    tex_wip.SetTextSize(0.0485); tex_wip.SetLineWidth(2)
    # tex_eot = r.TLatex(0.70, 0.92, "1 E13 EoT (8 GeV)")
    # tex_eot.SetNDC(); tex_eot.SetTextFont(52)
    # tex_eot.SetTextSize(0.03); tex_eot.SetLineWidth(2)
    tex_ldmx.Draw("SAME")
    tex_wip.Draw("SAME")
    # tex_eot.Draw("SAME")
    return tex_ldmx, tex_wip#, tex_eot


def create_legend(position, width=0.15, height=0.2):
    positions = {
        'upper left':    (0.2,  0.6,  0.2  + width, 0.6  + height),
        'upper middle':  (0.4,  0.65, 0.4  + width, 0.65 + height),
        'upper right':   (0.65, 0.5,  0.65 + width, 0.5  + height),
        'middle right':  (0.55, 0.7,  0.55 + width, 0.7  + height),
        'middle':        (0.45, 0.5,  0.45 + width, 0.5  + height),
        'lower left':    (0.1,  0.1,  0.1  + width, 0.1  + height),
        'lower middle':  (0.4,  0.4,  0.4  + width, 0.4  + height),
        'lower right':   (0.7,  0.4,  0.7  + width, 0.4  + height),
    }
    if position in positions:
        return r.TLegend(*positions[position])
    raise ValueError(f"Unknown legend position: {position}")


# ─────────────────────────────────────────────────────────────
#  ROC Plotting Functions
# ─────────────────────────────────────────────────────────────

def plot_roc_pyroot(bkg_disc, signal_dict, output_dir, legendPOS='lower right'):
    """
    Draw ROC curves for one BDT comparing multiple signal masses.
    Produces both logx and linear x-axis plots.
    AUC values shown ONLY on the linear plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    r.gROOT.SetBatch(True)

    roc_colors = [r.kGreen + 3, r.kViolet - 3, r.kOrange + 7, r.kAzure + 1,
                  r.kRed + 1, r.kCyan + 2, r.kMagenta + 1, r.kBlack]

    roc_data = []
    for label, sig_disc in signal_dict.items():
        y_true = np.concatenate([
            np.zeros(len(bkg_disc), dtype=int),
            np.ones(len(sig_disc), dtype=int)
        ])
        y_score = np.concatenate([bkg_disc, sig_disc])
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        roc_data.append((label, fpr, tpr, auc_val))

    for use_log, suffix, xlim in [(True, 'logx', (1e-7, 1.0)),
                                   (False, 'linear', (0.0, 0.01))]:
        can = r.TCanvas(f"roc_{suffix}", "ROC Analysis", 800, 660)
        r.gStyle.SetPadRightMargin(.15)
        r.gStyle.SetPadTopMargin(0.1)
        r.gStyle.SetPadBottomMargin(0.14)
        r.gStyle.SetPadLeftMargin(0.15)
        can.cd()
        can.SetGrid()
        if use_log:
            can.SetLogx()

        mg = r.TMultiGraph()
        graphs = []

        for i, (label, fpr, tpr, auc_val) in enumerate(roc_data):
            n = len(fpr)
            fpr_arr = array('d', fpr)
            tpr_arr = array('d', tpr)
            graph = r.TGraph(n, fpr_arr, tpr_arr)
            graph.SetLineWidth(2)
            graph.SetLineColor(roc_colors[i % len(roc_colors)])
            graph.SetLineStyle(1)
            # AUC only on linear plot
            if use_log:
                graph.SetTitle(f"{label}")
            else:
                graph.SetTitle(f"{label} (AUC={auc_val:.4f})")
            mg.Add(graph)
            graphs.append(graph)

        mg.Draw("AL")
        mg.SetTitle(";False Positive Rate (Background Efficiency);True Positive Rate (Signal Efficiency)")
        mg.GetXaxis().SetLimits(xlim[0], xlim[1])
        mg.SetMinimum(0)
        mg.SetMaximum(1.05)

        legend = create_legend(legendPOS)
        legend.SetTextSize(0.03)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        for g in graphs:
            legend.AddEntry(g, g.GetTitle(), "l")
        legend.Draw("SAME")

        tex = _draw_ldmx_header()

        outpath = os.path.join(output_dir, f"ROC_{suffix}.png")
        can.SaveAs(outpath)
        print(f"[ROC] Saved: {outpath}")


def plot_bdt_comparison_roc(bkg_disc_by_bdt, sig_disc_by_bdt,
                            mass_label, output_dir, legendPOS='lower right'):
    """
    Plot ROC curves comparing 2+ BDTs for a single mass point.
    Each BDT has its own background and signal disc values.
    AUC shown only on linear plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    r.gROOT.SetBatch(True)

    roc_colors = [r.kGreen + 3, r.kViolet - 3, r.kOrange + 7, r.kAzure + 1,
                  r.kRed + 1, r.kCyan + 2, r.kMagenta + 1, r.kBlack]

    roc_data = []
    for i, (bdt_label, sig_disc) in enumerate(sig_disc_by_bdt.items()):
        bkg_disc = bkg_disc_by_bdt[bdt_label]
        y_true = np.concatenate([
            np.zeros(len(bkg_disc), dtype=int),
            np.ones(len(sig_disc), dtype=int)
        ])
        y_score = np.concatenate([bkg_disc, sig_disc])
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        roc_data.append((bdt_label, fpr, tpr, auc_val))

    for use_log, suffix, xlim in [(True, 'logx', (1e-7, 1.0)),
                                   (False, 'linear', (0.0, 0.01))]:
        can = r.TCanvas(f"roc_bdt_cmp_{mass_label}_{suffix}", "BDT Comparison", 800, 660)
        r.gStyle.SetPadRightMargin(.15)
        r.gStyle.SetPadTopMargin(0.1)
        r.gStyle.SetPadBottomMargin(0.14)
        r.gStyle.SetPadLeftMargin(0.15)
        can.cd()
        can.SetGrid()
        if use_log:
            can.SetLogx()

        mg = r.TMultiGraph()
        graphs = []

        for i, (bdt_label, fpr, tpr, auc_val) in enumerate(roc_data):
            n = len(fpr)
            fpr_arr = array('d', fpr)
            tpr_arr = array('d', tpr)
            graph = r.TGraph(n, fpr_arr, tpr_arr)
            graph.SetLineWidth(2)
            graph.SetLineColor(roc_colors[i % len(roc_colors)])
            graph.SetLineStyle(1)
            if use_log:
                graph.SetTitle(f"{bdt_label}")
            else:
                graph.SetTitle(f"{bdt_label} (AUC={auc_val:.4f})")
            mg.Add(graph)
            graphs.append(graph)

        mg.Draw("AL")
        mg.SetTitle(";False Positive Rate (Background Efficiency);True Positive Rate (Signal Efficiency)")
        mg.GetXaxis().SetLimits(xlim[0], xlim[1])
        mg.SetMinimum(0)
        mg.SetMaximum(1.05)

        legend = create_legend(legendPOS)
        legend.SetTextSize(0.03)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetHeader(f"{mass_label} Signal", "C")
        for g in graphs:
            legend.AddEntry(g, g.GetTitle(), "l")
        legend.Draw("SAME")

        tex = _draw_ldmx_header()

        outpath = os.path.join(output_dir, f"ROC_bdt_comparison_{suffix}.png")
        can.SaveAs(outpath)
        print(f"[ROC BDT Comp] Saved: {outpath}")


def plot_combined_bdt_roc(bkg_disc_by_bdt, sig_disc_by_bdt,
                          bdt_branches, signal_masses,
                          output_dir, legendPOS='lower right'):
    """
    All masses × all BDTs on a single ROC canvas.
    Colors: each mass gets a base color; BDT 1 = darker shade, BDT 2 = lighter shade.
    AUC shown only on linear plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    r.gROOT.SetBatch(True)

    # Color pairs: (dark shade, light shade) using ROOT color wheel
    # 1 MeV = Red, 10 MeV = Green, 100 MeV = Blue, 1000 MeV = Orange
#     mass_color_pairs = [
#     (r.TColor.GetColor("#B22222"), r.TColor.GetColor("#F28E8E")),  # reds
#     (r.TColor.GetColor("#1B7F5A"), r.TColor.GetColor("#7BC8A4")),  # greens
#     (r.TColor.GetColor("#1F4E99"), r.TColor.GetColor("#7FA6E8")),  # blues
#     (r.TColor.GetColor("#A15C00"), r.TColor.GetColor("#F6B04E")),  # oranges
# ]
    mass_color_pairs = [(r.TColor.GetColor(0.0, 0.8, 0.0),   # light green 1 MeV
                        r.TColor.GetColor(0.0, 0.5, 0.0)),   # dark green 1 MeV
                (r.TColor.GetColor(0.8, 0.0, 0.8),   # light purple  10 MeV
                r.TColor.GetColor(0.5, 0.0, 0.5)),   # dark purple   10 MeV
                (r.TColor.GetColor(1.0, 0.6, 0.0),  # light orange  100 MeV
                r.TColor.GetColor(0.8, 0.3, 0.0)),   # dark orange   100 MeV
                (r.TColor.GetColor(0.3, 0.6, 1.0),   # light blue    1000 MeV
                r.TColor.GetColor(0.0, 0.3, 0.7) )   # dark blue     1000 MeV
    ]
       
    
    
    
    bdt_labels_list = list(bdt_branches.keys())

    # Compute ROC data
    roc_data = []
    for mass_idx, (mass_label, mass_key) in enumerate(signal_masses.items()):
        color_pair = mass_color_pairs[mass_idx % len(mass_color_pairs)]
        for bdt_idx, bdt_label in enumerate(bdt_labels_list):
            if mass_label not in sig_disc_by_bdt.get(bdt_label, {}):
                continue
            bkg_disc = bkg_disc_by_bdt[bdt_label]
            sig_disc = sig_disc_by_bdt[bdt_label][mass_label]
            y_true = np.concatenate([
                np.zeros(len(bkg_disc), dtype=int),
                np.ones(len(sig_disc), dtype=int)
            ])
            y_score = np.concatenate([bkg_disc, sig_disc])
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
            auc_val = metrics.auc(fpr, tpr)
            # BDT 1 (index 0) = darker, BDT 2 (index 1) = lighter
            color = color_pair[0] if bdt_idx == 0 else color_pair[1]
            legend_label = f"{mass_label} - {bdt_label}"
            roc_data.append((legend_label, fpr, tpr, auc_val, color))

    for use_log, suffix, xlim in [(True, 'logx', (1e-7, 1.0)),
                                   (False, 'linear', (0.0, 0.01))]:
        can = r.TCanvas(f"roc_combined_{suffix}", "Combined BDT Comparison", 800, 660)
        r.gStyle.SetPadRightMargin(.15)
        r.gStyle.SetPadTopMargin(0.1)
        r.gStyle.SetPadBottomMargin(0.14)
        r.gStyle.SetPadLeftMargin(0.15)
        can.cd()
        can.SetGrid()
        if use_log:
            can.SetLogx()

        mg = r.TMultiGraph()
        graphs = []

        for legend_label, fpr, tpr, auc_val, color in roc_data:
            n = len(fpr)
            fpr_arr = array('d', fpr)
            tpr_arr = array('d', tpr)
            graph = r.TGraph(n, fpr_arr, tpr_arr)
            graph.SetLineWidth(2)
            graph.SetLineColor(color)
            graph.SetLineStyle(1)
            if use_log:
                graph.SetTitle(legend_label)
            else:
                graph.SetTitle(f"{legend_label} (AUC={auc_val:.4f})")
            mg.Add(graph)
            graphs.append(graph)

        mg.Draw("AL")
        mg.SetTitle(";False Positive Rate (Background Efficiency);True Positive Rate (Signal Efficiency)")
        mg.GetXaxis().SetLimits(xlim[0], xlim[1])
        mg.SetMinimum(0)
        mg.SetMaximum(1.05)

        # Different legend positions for logx vs linear
        if use_log:
            # Bottom-right of the plot area (inside the axes)
            legend = r.TLegend(0.55, 0.15, 0.82, 0.45)
        else:
            legend = r.TLegend(0.4, 0.2, 0.4 + 0.15, 0.2 + 0.32)
        legend.SetTextSize(0.025)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        for g in graphs:
            legend.AddEntry(g, g.GetTitle(), "l")
        legend.Draw("SAME")

        tex = _draw_ldmx_header()

        outpath = os.path.join(output_dir, f"ROC_combined_bdt_{suffix}.png")
        can.SaveAs(outpath)
        print(f"[ROC Combined] Saved: {outpath}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    bdt_labels = list(BDT_BRANCHES.keys())

    print("=" * 60)
    print("  ROC CURVE ANALYSIS")
    print(f"  BDTs: {', '.join(bdt_labels)}")
    print("=" * 60)

    # ─── Load ALL background disc values once ───
    bkg_disc_by_bdt = {}
    for bdt_label, bdt_branch in BDT_BRANCHES.items():
        print(f"\n--- Loading background for {bdt_label} ---")
        bkg_container = SampleContainer(
            path=ECAL_PN_DIR,
            is_signal=False,
            ecalveto_branch=bdt_branch,
            max_files=MAX_FILES_BKG,
            take_last=TAKE_LAST_BKG,
        )
        bkg_disc_by_bdt[bdt_label] = bkg_container.get_disc_values()
        print(f"  Background ({bdt_label}): {len(bkg_disc_by_bdt[bdt_label])} events")

    # ─── Load ALL signal disc values once ───
    sig_disc_by_bdt = {bl: {} for bl in BDT_BRANCHES}
    for mass_label, mass_key in SIGNAL_MASSES.items():
        pattern = os.path.join(SIGNAL_DIR, f"*{mass_key}*")
        matching = sorted(glob.glob(pattern))
        if not matching:
            print(f"[WARNING] No signal files for {mass_label}")
            continue

        for bdt_label, bdt_branch in BDT_BRANCHES.items():
            sig_container = SampleContainer(
                path=SIGNAL_DIR,
                is_signal=True,
                ecalveto_branch=bdt_branch,
                trigger_branch=TRIGGER_BRANCH,
                max_files=MAX_FILES_SIG,
            )
            if MAX_FILES_SIG is not None:
                sig_container.files = matching[:MAX_FILES_SIG]
            else:
                sig_container.files = matching

            disc = sig_container.get_disc_values()
            sig_disc_by_bdt[bdt_label][mass_label] = disc
            print(f"  {mass_label} ({bdt_label}): {len(disc)} events")

    # ─── Process explicit pairs of BDTs ───
    for i, j in BDT_PAIRS:
        if i < 1 or j < 1 or i > len(bdt_labels) or j > len(bdt_labels):
            print(f"[WARNING] Skipping invalid pair ({i}, {j}) — out of range")
            continue
        pair = [bdt_labels[i - 1], bdt_labels[j - 1]]
        pair_branches = {bl: BDT_BRANCHES[bl] for bl in pair}

        comparison_tag = "_vs_".join(bl.replace(" ", "_") for bl in pair)
        output_dir = os.path.join(OUTPUT_BASE, comparison_tag, "roc_curves")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Pair: {' vs '.join(pair)}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")

        # ─── Per-mass BDT comparison ROC plots (only if pair has 2) ───
        if len(pair) >= 2:
            print("\n--- Per-mass BDT comparison ROC plots ---")
            for mass_label, mass_key in SIGNAL_MASSES.items():
                has_data = all(mass_label in sig_disc_by_bdt[bl] for bl in pair)
                if not has_data:
                    print(f"[WARNING] Skipping {mass_label} — missing data")
                    continue

                mass_output_dir = os.path.join(output_dir, f"bdt_comparison_{mass_key}")
                plot_bdt_comparison_roc(
                    bkg_disc_by_bdt={bl: bkg_disc_by_bdt[bl] for bl in pair},
                    sig_disc_by_bdt={bl: sig_disc_by_bdt[bl][mass_label] for bl in pair},
                    mass_label=mass_label,
                    output_dir=mass_output_dir,
                )

        # ─── All-masses ROC per BDT (in this pair) ───
        print("\n--- All-masses ROC per BDT ---")
        for bdt_label in pair:
            bkg_disc = bkg_disc_by_bdt[bdt_label]
            sig_dict = sig_disc_by_bdt[bdt_label]
            if len(bkg_disc) > 0 and sig_dict:
                bdt_output_dir = os.path.join(output_dir, f"roc_{bdt_label.replace(' ', '_')}")
                plot_roc_pyroot(bkg_disc, sig_dict, bdt_output_dir)

        # ─── Combined: all masses × pair BDTs (lighter/darker colors) ───
        if len(pair) >= 2:
            print("\n--- Combined all-masses × pair-BDTs ROC ---")
            plot_combined_bdt_roc(
                bkg_disc_by_bdt={bl: bkg_disc_by_bdt[bl] for bl in pair},
                sig_disc_by_bdt={bl: sig_disc_by_bdt[bl] for bl in pair},
                bdt_branches=pair_branches,
                signal_masses=SIGNAL_MASSES,
                output_dir=output_dir,
            )

    print(f"\n[DONE] All ROC plots saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    import json
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        cfg = json.loads(sys.argv[2])
        if "BDT_BRANCHES" in cfg:
            BDT_BRANCHES.clear()
            BDT_BRANCHES.update(cfg["BDT_BRANCHES"])
        if "ECAL_PN_DIR" in cfg:
            ECAL_PN_DIR = cfg["ECAL_PN_DIR"]
        if "SIGNAL_DIR" in cfg:
            SIGNAL_DIR = cfg["SIGNAL_DIR"]
        if "MAX_FILES_BKG" in cfg:
            MAX_FILES_BKG = cfg["MAX_FILES_BKG"]
        if "MAX_FILES_SIG" in cfg:
            MAX_FILES_SIG = cfg["MAX_FILES_SIG"]
        if "OUTPUT_BASE" in cfg:
            OUTPUT_BASE = cfg["OUTPUT_BASE"]
    main()

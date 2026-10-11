#!/usr/bin/env python3
"""
plot_disc_comparison.py
=======================
Compare BDT discriminator distributions between 2 (or 1) BDTs.
Plots both linear disc (0-1) and -log10(1-disc) (0-7).

Usage:
    python plot_disc_comparison.py
    (Edit the CONFIGURATION section below before running.)
"""

import os
import sys
import glob
import numpy as np
import awkward as ak
import uproot
from array import array
import ROOT as r

# Increase recursion limit for deeply nested ROOT file branch structures
sys.setrecursionlimit(10000)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────

# --- BDTs to compare ---
# To compare 2 BDTs, put both entries.  To plot only 1 BDT, comment out one.
BDT_BRANCHES = {
    "BDT Truth":  "EcalVeto_SegMip_Truth_BDT",
    "BDT Recoil":      "EcalVeto_SegMip_Recoil_BDT",
}

# --- Input directories ---
BASE_SAMPLES = "/home/jihoonyoo/LDMX/8GeV_Samples/roc"
ECAL_PN_DIR  = os.path.join(BASE_SAMPLES, "ecal_pn_SP_recoil_small_roc_samples")
SIGNAL_DIR   = os.path.join(BASE_SAMPLES, "signal_small_recoil_small_roc_samples")

# --- Signal mass points ---
SIGNAL_MASSES = {
    "1 MeV":    "1MeV",
    "10 MeV":   "10MeV",
    "100 MeV":  "100MeV",
    "1000 MeV": "1000MeV",
}

# --- Output directory ---
# A subdirectory named after the BDTs being compared will be created automatically.
OUTPUT_BASE = "/home/jihoonyoo/LDMX/v4.5.11/ldmx-sw/bdt_analysis/BDT_Comparisons"

# --- File limits (for quick testing, set to None for all files) ---
MAX_FILES_BKG     = 100
MAX_FILES_SIG     = 40
TAKE_LAST_BKG     = True   # Use last N bkg files (avoid training overlap)

# --- Trigger branch for signal ---
TRIGGER_BRANCH = "Trigger_sim"


# ─────────────────────────────────────────────────────────────
#  SampleContainer: loads ROOT files and extracts disc values
# ─────────────────────────────────────────────────────────────

class SampleContainer:
    """
    Loads .root files from a directory (recursively) or a single file,
    extracts the BDT discriminator values from the EcalVeto branch.
    """
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
        """Load disc values. For signal, only keep events where the trigger passed."""
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
    """Draw the standard LDMX / Work in Progress header."""
    tex_ldmx = r.TLatex(0.15, 0.92, "LDMX")
    tex_ldmx.SetNDC(); tex_ldmx.SetTextFont(61)
    tex_ldmx.SetTextSize(0.0675); tex_ldmx.SetLineWidth(2)

    tex_wip = r.TLatex(0.33, 0.92, "Work in Progress")
    tex_wip.SetNDC(); tex_wip.SetTextFont(52)
    tex_wip.SetTextSize(0.0485); tex_wip.SetLineWidth(2)

    tex_eot = r.TLatex(0.70, 0.92, "1 E14 EoT (8 GeV)")
    tex_eot.SetNDC(); tex_eot.SetTextFont(52)
    tex_eot.SetTextSize(0.03); tex_eot.SetLineWidth(2)

    tex_ldmx.Draw("SAME"); tex_wip.Draw("SAME"); tex_eot.Draw("SAME")
    return tex_ldmx, tex_wip, tex_eot


def create_legend(position, width=0.15, height=0.2):
    """Create a TLegend at a named position."""
    positions = {
        'upper left':    (0.17, 0.68, 0.17 + width, 0.68 + height),  # top-left INSIDE the frame
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


def _fill_th1_from_array(h, values):
    """Fill a TH1F efficiently from a numpy array."""
    nbins = h.GetNbinsX()
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    counts, _ = np.histogram(values, bins=nbins, range=(xmin, xmax))
    for ib in range(nbins):
        h.SetBinContent(ib + 1, float(counts[ib]))
    overflow = np.sum(values >= xmax)
    h.SetBinContent(nbins + 1, float(overflow))
    h.SetEntries(len(values))


def drawMultHists(hists, labels, xtitle, filename, title,
                  scale=False, log=False, ytitle='Events',
                  legendPOS='upper left', sufx=''):
    """Overlay multiple TH1s on one canvas with LDMX header."""
    assert len(hists) > 0, "No histograms provided"
    assert len(hists) == len(labels), "len(hists) must equal len(labels)"

    r.gROOT.SetBatch(True)
    can = r.TCanvas(f"canvas_{os.path.basename(filename)}", "", 800, 800)
    r.gStyle.SetPadRightMargin(.15)
    r.gStyle.SetPadTopMargin(0.1)
    r.gStyle.SetPadBottomMargin(0.14)
    r.gStyle.SetPadLeftMargin(0.15)
    can.cd()
    if log:
        can.SetLogy()

    drawn_hists = []
    for h in hists:
        hh = h.Clone()
        if scale:
            n = hh.GetEntries()
            if n and n > 0:
                hh.Scale(1.0 / float(n))
        drawn_hists.append(hh)

    h0 = drawn_hists[0]
    h0.GetXaxis().SetTitle(xtitle)
    h0.GetYaxis().SetTitle(ytitle)

    ymax = 0.0
    ymin_pos = None
    for hh in drawn_hists:
        ymax = max(ymax, hh.GetMaximum())
        if log:
            for ib in range(1, hh.GetNbinsX() + 1):
                c = hh.GetBinContent(ib)
                if c > 0.0:
                    ymin_pos = c if ymin_pos is None else min(ymin_pos, c)
    if log:
        if ymin_pos is None:
            ymin_pos = 1e-3
        h0.SetMinimum(0.5 * ymin_pos)
        h0.SetMaximum(10.0 * ymax if ymax > 0 else 1.0)
    else:
        h0.SetMinimum(0.0)
        h0.SetMaximum(1.2 * ymax if ymax > 0 else 1.0)

    colors = [r.kCyan + 2, r.kRed + 1, r.kBlue + 1, r.kGreen + 2,
              r.kMagenta + 1, r.kOrange + 7, r.kViolet + 2, r.kBlack]

    for i, hh in enumerate(drawn_hists):
        hh.SetLineColor(colors[i % len(colors)])
        hh.SetMarkerColor(colors[i % len(colors)])
        hh.SetLineWidth(2)
        hh.SetStats(0)
        nbins = hh.GetNbinsX()
        overflow = hh.GetBinContent(nbins + 1)
        hh.AddBinContent(nbins, overflow)
        hh.Draw("HIST" if i == 0 else "HIST SAME")

    tex = _draw_ldmx_header()

    legend = create_legend(legendPOS)
    legend.SetLineColor(0)
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    for hh, lab in zip(drawn_hists, labels):
        legend.AddEntry(hh, lab, "l")
    legend.Draw("SAME")

    can.SaveAs(f"{filename}{sufx}.png")
    print(f"[PLOT] Saved: {filename}{sufx}.png")
    return can, tex, legend, drawn_hists


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    # Build output directory name from BDT labels
    bdt_labels = list(BDT_BRANCHES.keys())
    comparison_tag = "_vs_".join(bl.replace(" ", "_") for bl in bdt_labels)
    output_dir = os.path.join(OUTPUT_BASE, comparison_tag, "disc_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  DISCRIMINATOR DISTRIBUTION COMPARISON")
    print(f"  BDTs: {', '.join(bdt_labels)}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # ─── Load background disc values per BDT ───
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

    # ─── Load signal disc values per BDT per mass ───
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

    # ─── Plot: Per-BDT disc distributions (bkg + all signal masses) ───
    print("\n" + "=" * 60)
    print("  PER-BDT DISC DISTRIBUTIONS (bkg + signal per mass)")
    print("=" * 60)

    for bdt_label in BDT_BRANCHES:
        bkg_disc = bkg_disc_by_bdt[bdt_label]
        bdt_tag = bdt_label.replace(' ', '_')

        disc_hists = []
        disc_log_hists = []
        hist_labels = []

        # Background
        h_bkg = r.TH1F(f"disc_bkg_{bdt_tag}", "", 100, 0, 1)
        _fill_th1_from_array(h_bkg, bkg_disc)
        h_bkg_log = r.TH1F(f"disc_log_bkg_{bdt_tag}", "", 100, 0, 7)
        mask = bkg_disc < 1.0
        neglog_bkg = -np.log10(1.0 - bkg_disc[mask])
        _fill_th1_from_array(h_bkg_log, neglog_bkg)

        disc_hists.append(h_bkg)
        disc_log_hists.append(h_bkg_log)
        hist_labels.append("Ecal PN (bkg)")

        # Signal per mass
        for mass_label in SIGNAL_MASSES:
            if mass_label not in sig_disc_by_bdt.get(bdt_label, {}):
                continue
            sig_disc = sig_disc_by_bdt[bdt_label][mass_label]

            h_sig = r.TH1F(f"disc_sig_{mass_label.replace(' ','_')}_{bdt_tag}", "", 100, 0, 1)
            _fill_th1_from_array(h_sig, sig_disc)
            h_sig_log = r.TH1F(f"disc_log_sig_{mass_label.replace(' ','_')}_{bdt_tag}", "", 100, 0, 7)
            mask_s = sig_disc < 1.0
            neglog_sig = -np.log10(1.0 - sig_disc[mask_s])
            _fill_th1_from_array(h_sig_log, neglog_sig)

            disc_hists.append(h_sig)
            disc_log_hists.append(h_sig_log)
            hist_labels.append(f"{mass_label} Signal")

        # Linear disc plot
        drawMultHists(
            hists=disc_hists,
            labels=hist_labels,
            title=f"{bdt_label} Discriminator",
            xtitle="BDT Discriminator Score",
            ytitle="Normalized Events",
            filename=os.path.join(output_dir, f"disc_linear_{bdt_tag}"),
            log=True,
            legendPOS='upper left',
            scale=True,
        )

        # -log10(1-disc) plot
        drawMultHists(
            hists=disc_log_hists,
            labels=hist_labels,
            title=f"{bdt_label} Discriminator",
            xtitle="-log_{10}(1 - BDT Disc Score)",
            ytitle="Normalized Events",
            filename=os.path.join(output_dir, f"disc_neglog_{bdt_tag}"),
            log=True,
            legendPOS='upper left',
            scale=True,
        )

    # ─── Plot: Side-by-side BDT comparison per sample ───
    # Only if we have 2 BDTs
    if len(BDT_BRANCHES) >= 2:
        print("\n" + "=" * 60)
        print("  SIDE-BY-SIDE BDT COMPARISON PER SAMPLE")
        print("=" * 60)

        # Background comparison
        comp_hists_lin = []
        comp_hists_log = []
        comp_labels = []
        for bdt_label in BDT_BRANCHES:
            bkg_disc = bkg_disc_by_bdt[bdt_label]
            bdt_tag = bdt_label.replace(' ', '_')

            h = r.TH1F(f"comp_bkg_lin_{bdt_tag}", "", 100, 0, 1)
            _fill_th1_from_array(h, bkg_disc)
            comp_hists_lin.append(h)

            h_log = r.TH1F(f"comp_bkg_log_{bdt_tag}", "", 100, 0, 7)
            mask = bkg_disc < 1.0
            _fill_th1_from_array(h_log, -np.log10(1.0 - bkg_disc[mask]))
            comp_hists_log.append(h_log)

            comp_labels.append(f"Ecal PN - {bdt_label}")

        drawMultHists(
            hists=comp_hists_lin, labels=comp_labels,
            title="Background BDT Comparison",
            xtitle="BDT Discriminator Score", ytitle="Normalized Events",
            filename=os.path.join(output_dir, f"bdt_comp_bkg_linear_{comparison_tag}"),
            log=True, scale=True, legendPOS='upper left',
        )
        drawMultHists(
            hists=comp_hists_log, labels=comp_labels,
            title="Background BDT Comparison",
            xtitle="-log_{10}(1 - BDT Disc Score)", ytitle="Normalized Events",
            filename=os.path.join(output_dir, f"bdt_comp_bkg_neglog_{comparison_tag}"),
            log=True, scale=True, legendPOS='upper left',
        )

        # Per-mass signal comparison
        for mass_label, mass_key in SIGNAL_MASSES.items():
            comp_hists_lin = []
            comp_hists_log = []
            comp_labels = []
            skip = False
            for bdt_label in BDT_BRANCHES:
                if mass_label not in sig_disc_by_bdt.get(bdt_label, {}):
                    skip = True
                    break
                sig_disc = sig_disc_by_bdt[bdt_label][mass_label]
                bdt_tag = bdt_label.replace(' ', '_')
                mlabel_tag = mass_label.replace(' ', '_')

                h = r.TH1F(f"comp_{mlabel_tag}_lin_{bdt_tag}", "", 100, 0, 1)
                _fill_th1_from_array(h, sig_disc)
                comp_hists_lin.append(h)

                h_log = r.TH1F(f"comp_{mlabel_tag}_log_{bdt_tag}", "", 100, 0, 7)
                mask_s = sig_disc < 1.0
                _fill_th1_from_array(h_log, -np.log10(1.0 - sig_disc[mask_s]))
                comp_hists_log.append(h_log)

                comp_labels.append(f"{mass_label} - {bdt_label}")

            if skip:
                continue

            drawMultHists(
                hists=comp_hists_lin, labels=comp_labels,
                title=f"{mass_label} Signal BDT Comparison",
                xtitle="BDT Discriminator Score", ytitle="Normalized Events",
                filename=os.path.join(output_dir, f"bdt_comp_{mass_key}_linear_{comparison_tag}"),
                log=True, scale=True, legendPOS='upper left',
            )
            drawMultHists(
                hists=comp_hists_log, labels=comp_labels,
                title=f"{mass_label} Signal BDT Comparison",
                xtitle="-log_{10}(1 - BDT Disc Score)", ytitle="Normalized Events",
                filename=os.path.join(output_dir, f"bdt_comp_{mass_key}_neglog_{comparison_tag}"),
                log=True, scale=True, legendPOS='upper left',
            )

    print(f"\n[DONE] All disc comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    import json
    # Allow overriding config via --config '{"key": "value"}' CLI arg
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

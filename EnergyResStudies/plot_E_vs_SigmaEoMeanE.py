import ROOT

ROOT.gStyle.SetPadRightMargin(0.03)

# Create TGraphErrors
graph = ROOT.TGraphErrors("energy_resolution.txt", "%lg %lg %lg" )

# Define the fit function f(E) = s/sqrt(E) + n/E + c
fit_func = ROOT.TF1("fit_func", "[0]/sqrt(x) + [1]/x + [2]", 0.04, 4.5)
fit_func.SetParameters(6, 0.01, 0.1)  # Initial guess

# Perform the fit
graph.Fit(fit_func, "WR")

# Extract fit parameters and uncertainties
s_fit, s_err = fit_func.GetParameter(0), fit_func.GetParError(0)
n_fit, n_err = fit_func.GetParameter(1), fit_func.GetParError(1)
c_fit, c_err = fit_func.GetParameter(2), fit_func.GetParError(2)

# Set graph styles
graph.SetTitle("Energy Resolution vs Injected Energy")
graph.GetXaxis().SetTitle("Injected Energy (GeV)")
graph.GetYaxis().SetTitle("#sigma_{E} / E_{avg}")
graph.SetMarkerStyle(20)
graph.SetMarkerSize(1)
graph.SetMarkerColor(ROOT.kBlue)



interval = ROOT.TGraphErrors(graph.GetN())
for i in range(graph.GetN()):
    interval.SetPoint(i, graph.GetX()[i], 0)
ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(interval, 0.95)

# Create canvas and draw graph
canvas = ROOT.TCanvas("canvas", "Energy Resolution Fit", 800, 800)
graph.Draw("AP")
interval.SetFillColorAlpha(ROOT.kRed-7, 0.3)
# interval.SetFillColor(ROOT.kRed-7)
interval.Draw("3same")
fit_func.Draw("psame")

# Display fit results on the plot
fit_text = ROOT.TLatex()
fit_text.SetNDC()  # Normalized device coordinates
fit_text.SetTextSize(0.03)
fit_text.SetTextAlign(13)  # Left alignment

# Fit function parameters (Fix ± issue with #pm)
fit_text.DrawLatex(0.15, 0.85, f"Fit: s/#sqrt{{E}} + n/E + c")
fit_text.DrawLatex(0.15, 0.80, f"s = {s_fit:.4f} #pm {s_err:.4f}")
fit_text.DrawLatex(0.15, 0.75, f"n = {n_fit:.4f} #pm {n_err:.4f}")
fit_text.DrawLatex(0.15, 0.70, f"c = {c_fit:.4f} #pm {c_err:.4f}")

# Additional information (photon gun, 0 degree, no noise)
fit_text.SetTextSize(0.035)
fit_text.SetTextColor(ROOT.kBlue)
fit_text.DrawLatex(0.60, 0.85, "Photon gun, 0^{#circ}")  # Fix degree symbol
fit_text.DrawLatex(0.60, 0.80, "No noise")

# Save and show the plot
canvas.SaveAs("energy_resolution_fit_fixed.png")
canvas.Draw()


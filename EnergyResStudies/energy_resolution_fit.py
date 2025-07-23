import ROOT

ROOT.gStyle.SetPadRightMargin(0.03)

# Create TGraphErrors
graph = ROOT.TGraphErrors("energy_analysis.txt", "%lg %lg %lg %lg" )

# Define the fit function f(E) = s/sqrt(E) + n/E + c
# fit_func = ROOT.TF1("fit_func", "[0]/sqrt(x) + [1]/x + [2]", 0.04, 4.5)
# instead of direct sum, use quadratic sum
# fit_func = ROOT.TF1("fit_func", "sqrt([0]^2/x + [1]^2/(x*x) + [2]^2)", 0.04, 4.5)
# fit_func.SetParameters(0.17, 0.01, 0.1)  # Initial guess

def fittingFunction(x, par):
    part1 = par[0] / ROOT.TMath.Sqrt(x[0])
    part3 = par[1] / x[0]
    part2 = par[2]
    sum_of_squares = ROOT.TMath.Sqrt(part1**2 + part2**2 + part3**2)
    return sum_of_squares

fit_func = ROOT.TF1("fit_func", fittingFunction, 0.05, 5.0, 3)
fit_func.SetParameters(0.2135, 0.00001, 0.02231)  # Initial guess

# Perform the fit
graph.Fit(fit_func, "MR", "", 0.05, 8.5)

# Extract fit parameters and uncertainties
s_fit, s_err = fit_func.GetParameter(0), fit_func.GetParError(0)
n_fit, n_err = fit_func.GetParameter(1), fit_func.GetParError(1)
c_fit, c_err = fit_func.GetParameter(2), fit_func.GetParError(2)

# Set graph styles
graph.SetTitle(" ")
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
interval.SetFillStyle(3001)
interval.Draw("3same")
fit_func.Draw("psame")

# Display fit results on the plot
fit_text = ROOT.TLatex()
fit_text.SetNDC()  # Normalized device coordinates
fit_text.SetTextSize(0.03)
fit_text.SetTextAlign(13)  # Left alignment

# Fit function parameters (Fix ± issue with #pm)
fit_text.DrawLatex(0.60, 0.70, "Fit: #sqrt{(s/#sqrt{E})^{2} + (n/E)^{2} + c^{2}}")
fit_text.DrawLatex(0.60, 0.65, f"s = {s_fit:.4f} #pm {s_err:.4f}")
fit_text.DrawLatex(0.60, 0.60, f"n = {n_fit:.4f} #pm {n_err:.4f}")
fit_text.DrawLatex(0.60, 0.55, f"c = {c_fit:.4f} #pm {c_err:.4f}")

# Additional information (photon gun, 0 degree, no noise)
fit_text.SetTextSize(0.035)
fit_text.SetTextColor(ROOT.kBlue)
fit_text.DrawLatex(0.60, 0.85, "Photon gun, 0^{#circ}")
fit_text.DrawLatex(0.60, 0.80, "with noise sigma = 1.6")
fit_text.DrawLatex(0.60, 0.75, "noise threshold = 58")

# Save and show the plot
canvas.SaveAs("energy_resolution_fit.png")
canvas.Draw()

mport ROOT

# Open the ROOT file
file = ROOT.TFile.Open("file_path/hist.root")
hist = file.Get("EcalDigiVerify/EcalDigiVerify_total_rec_energy")

# Check if histogram is loaded
if not hist:
    print("Error: Histogram 'EcalDigiVerify_total_rec_energy' not found in EcalDigiVerify!")
    exit()

# Define Gaussian fit function with updated rangeconda activate root-env
gauss_fit = ROOT.TF1("gauss_fit", "gaus", -3.4155, 4.0165)
hist.Fit(gauss_fit, "R")

# Get fit parameters and uncertainties
mean = gauss_fit.GetParameter(1)  # Mean (μ)
mean_err = gauss_fit.GetParError(1)  # Uncertainty on Mean (Δμ)
sigma = gauss_fit.GetParameter(2)  # Sigma (σ)
sigma_err = gauss_fit.GetParError(2)  # Uncertainty on Sigma (Δσ)
chi2 = gauss_fit.GetChisquare()
ndf = gauss_fit.GetNDF()

# Create canvas
canvas = ROOT.TCanvas("canvas", "Fit Result", 800, 600)
canvas.cd()
hist.Draw("E")
gauss_fit.SetLineColor(ROOT.kRed)
gauss_fit.Draw("SAME")

# Text box with updated values
stats = ROOT.TPaveText(0.6, 0.4, 0.88, 0.6, "NDC")
stats.AddText(f"Mean = {mean:.2f} #pm {mean_err:.2f} MeV")
stats.AddText(f"Sigma = {sigma:.2f} #pm {sigma_err:.2f} MeV")
stats.AddText(f"#chi^{{2}}/NDF = {chi2:.2f} / {ndf}")
stats.SetFillColor(0)
stats.Draw("SAME")

# Save as png
canvas.SaveAs("gaussian_fit.png")

# Close the ROOT file
file.Close()

# Display canvas
canvas.Draw()


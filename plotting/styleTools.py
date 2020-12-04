import ROOT as rt
import math
from array import array

color_comp1=634    # kRed+2
color_comp2=862    # kAzure+2
color_comp3=797    # kOrange-3
color_comp4=882    # kViolet+2
color_comp5=419    # kGreen+3
color_comp6=603    # kBlue+3
color_comp7=802    # kOrange+2
color_comp8=616    # kMagenta
color_comp9=600    # kBlue
color_comp10=434   # kCyan+2
color_comp11=800   # kOrange
color_comp12=417   # kGreen+1
color_comp13=632   # kRed

colors = {}
colors['color_comp1'] = color_comp1
colors['color_comp2'] = color_comp2
colors['color_comp3'] = color_comp3
colors['color_comp4'] = color_comp4
colors['color_comp5'] = color_comp5
colors['color_comp6'] = color_comp6
colors['color_comp7'] = color_comp7
colors['color_comp8'] = color_comp8
colors['color_comp9'] = color_comp9
colors['color_comp10'] = color_comp10
colors['color_comp11'] = color_comp11
colors['color_comp12'] = color_comp12
colors['color_comp13'] = color_comp13


def MakeCanvas(name,title,dX=500,dY=500):
  #Start with a canvas
  canvas = rt.TCanvas(name,title,0,0,dX,dY)

  #General overall stuff
  canvas.SetFillColor      (0)
  canvas.SetBorderMode     (0)
  canvas.SetBorderSize     (10)

  #Set margins to reasonable defaults
  canvas.SetLeftMargin     (0.18)
  canvas.SetRightMargin    (0.05)
  canvas.SetTopMargin      (0.08)
  canvas.SetBottomMargin   (0.15)
  
  #Setup a frame which makes sense
  canvas.SetFrameFillStyle (0)
  canvas.SetFrameLineStyle (0)
  canvas.SetFrameLineWidth (3)
  canvas.SetFrameBorderMode(0)
  canvas.SetFrameBorderSize(10)
  canvas.SetFrameFillStyle (0)
  canvas.SetFrameLineStyle (0)
  canvas.SetFrameBorderMode(0)
  canvas.SetFrameBorderSize(10)

  return canvas

def InitHist(hist, xtit, ytit, color, style):
  hist.SetXTitle(xtit)
  hist.SetYTitle(ytit)
  hist.SetLineColor(rt.kBlack)
  hist.SetLineWidth(    3.)
  hist.SetFillColor(color )
  hist.SetFillStyle(style )
  hist.SetTitleSize  (0.055,'Y')
  hist.SetTitleOffset(1.600,'Y')
  hist.SetLabelOffset(0.014,'Y')
  hist.SetLabelSize  (0.040,'Y')
  hist.SetLabelFont  (62   ,'Y')
  hist.SetTitleSize  (0.055,'X')
  hist.SetTitleOffset(1.300,'X')
  hist.SetLabelOffset(0.014,'X')
  hist.SetLabelSize  (0.040,'X')
  hist.SetLabelFont  (62   ,'X')
  hist.SetMarkerStyle(20)
  hist.SetMarkerColor(color)
  hist.SetMarkerSize (1.3)
  hist.GetYaxis().SetTitleFont(62)
  hist.GetXaxis().SetTitleFont(62)
  hist.SetTitle('')  

def addOverFlow(h):
  nbins = h.GetNbinsX()+1

  e1 = h.GetBinError(nbins-1)
  e2 = h.GetBinError(nbins)
  h.AddBinContent(nbins-1, h.GetBinContent(nbins))
  h.SetBinError(nbins-1, math.sqrt(e1*e1 + e2*e2))
  h.SetBinContent(nbins, 0)
  h.SetBinError(nbins, 0)
  return h

def SetLegendStyle(leg):
  leg.SetFillStyle (0)
  leg.SetFillColor (0)
  leg.SetBorderSize(0)
  leg.SetTextSize(0.05)

def SetupColors():
  num = 5
  bands = 255
  colors = array('i',[])
  stops = array('d',[0.00, 0.34, 0.61, 0.84, 1.00])
  red = array('d',[0.50, 0.50, 1.00, 1.00, 1.00])
  green = array('d',[0.50, 1.00, 1.00, 0.60, 0.50])
  blue = array('d',[1.00, 1.00, 0.50, 0.40, 0.50])
  fi = rt.TColor.CreateGradientColorTable(num,stops,red,green,blue,bands)
  for i in range(bands):
    colors.append(fi+i)

  rt.gStyle.SetNumberContours(bands)
  rt.gStyle.SetPalette(bands, colors)

def SetStyle():
  MyStyle = rt.TStyle('New Style','Better than the default style :-)')
  rt.gStyle = MyStyle

  #Canvas
  MyStyle.SetCanvasColor     (0)
  MyStyle.SetCanvasBorderSize(10)
  MyStyle.SetCanvasBorderMode(0)
  MyStyle.SetCanvasDefH      (700)
  MyStyle.SetCanvasDefW      (700)
  MyStyle.SetCanvasDefX      (100)
  MyStyle.SetCanvasDefY      (100)

  #color palette for 2D temperature plots
  MyStyle.SetPalette(1,0)

  #Pads
  MyStyle.SetPadColor       (0)
  MyStyle.SetPadBorderSize  (10)
  MyStyle.SetPadBorderMode  (0)
  MyStyle.SetPadBottomMargin(0.13)
  MyStyle.SetPadTopMargin   (0.08)
  MyStyle.SetPadLeftMargin  (0.15)
  MyStyle.SetPadRightMargin (0.05)
  MyStyle.SetPadGridX       (0)
  MyStyle.SetPadGridY       (0)
  MyStyle.SetPadTickX       (1)
  MyStyle.SetPadTickY       (1)

  #Frames
  MyStyle.SetLineWidth(3)
  MyStyle.SetFrameFillStyle ( 0)
  MyStyle.SetFrameFillColor ( 0)
  MyStyle.SetFrameLineColor ( 1)
  MyStyle.SetFrameLineStyle ( 0)
  MyStyle.SetFrameLineWidth ( 3)
  MyStyle.SetFrameBorderSize(10)
  MyStyle.SetFrameBorderMode( 0)

  #Histograms
  MyStyle.SetHistFillColor(2)
  MyStyle.SetHistFillStyle(0)
  MyStyle.SetHistLineColor(1)
  MyStyle.SetHistLineStyle(0)
  MyStyle.SetHistLineWidth(3)
  MyStyle.SetNdivisions(505, 'X')

  #Functions
  MyStyle.SetFuncColor(1)
  MyStyle.SetFuncStyle(0)
  MyStyle.SetFuncWidth(2)

  #Various
  MyStyle.SetMarkerStyle(20)
  MyStyle.SetMarkerColor(rt.kBlack)
  MyStyle.SetMarkerSize (1.4)

  MyStyle.SetTitleBorderSize(0)
  MyStyle.SetTitleFillColor (0)
  MyStyle.SetTitleX         (0.2)

  MyStyle.SetTitleSize  (0.055,'X')
  MyStyle.SetTitleOffset(1.200,'X')
  MyStyle.SetLabelOffset(0.005,'X')
  MyStyle.SetLabelSize  (0.040,'X')
  MyStyle.SetLabelFont  (62   ,'X')

  MyStyle.SetStripDecimals(False)
  MyStyle.SetLineStyleString(11,'20 10')

  MyStyle.SetTitleSize  (0.055,'Y')
  MyStyle.SetTitleOffset(1.600,'Y')
  MyStyle.SetLabelOffset(0.010,'Y')
  MyStyle.SetLabelSize  (0.040,'Y')
  MyStyle.SetLabelFont  (62   ,'Y')

  MyStyle.SetTextSize   (0.055)
  MyStyle.SetTextFont   (62)

  MyStyle.SetStatFont   (62)
  MyStyle.SetTitleFont  (62)
  MyStyle.SetTitleFont  (62,'X')
  MyStyle.SetTitleFont  (62,'Y')

  MyStyle.SetOptStat    (0)

def SetTDRStyle():
  #Copied from https://ghm.web.cern.ch/ghm/plots/MacroExample/tdrstyle.C
  tdrStyle = rt.TStyle('tdrStyle','Style for P-TDR')

  #For the canvas:
  tdrStyle.SetCanvasBorderMode(0)
  tdrStyle.SetCanvasColor(rt.kWhite)
  tdrStyle.SetCanvasDefH(600) #Height of canvas
  tdrStyle.SetCanvasDefW(600) #Width of canvas
  tdrStyle.SetCanvasDefX(0)   #POsition on screen
  tdrStyle.SetCanvasDefY(0)

  #For the Pad:
  tdrStyle.SetPadBorderMode(0)
  #tdrStyle.SetPadBorderSize(Width_t size = 1)
  tdrStyle.SetPadColor(rt.kWhite)
  tdrStyle.SetPadGridX(False)
  tdrStyle.SetPadGridY(False)
  tdrStyle.SetGridColor(0)
  tdrStyle.SetGridStyle(3)
  tdrStyle.SetGridWidth(1)

  #For the frame:
  tdrStyle.SetFrameBorderMode(0)
  tdrStyle.SetFrameBorderSize(1)
  tdrStyle.SetFrameFillColor(0)
  tdrStyle.SetFrameFillStyle(0)
  tdrStyle.SetFrameLineColor(1)
  tdrStyle.SetFrameLineStyle(1)
  tdrStyle.SetFrameLineWidth(1)

  #For the histo:
  #tdrStyle.SetHistFillColor(1)
  #tdrStyle.SetHistFillStyle(0)
  tdrStyle.SetHistLineColor(1)
  tdrStyle.SetHistLineStyle(0)
  tdrStyle.SetHistLineWidth(1)
  #tdrStyle.SetLegoInnerR(Float_t rad = 0.5)
  #tdrStyle.SetNumberContours(Int_t number = 20)

  tdrStyle.SetEndErrorSize(2)
  #tdrStyle.SetErrorMarker(20)
  #tdrStyle.SetErrorX(0.)

  tdrStyle.SetMarkerStyle(20)

  #For the fit/function:
  tdrStyle.SetOptFit(1)
  tdrStyle.SetFitFormat('5.4g')
  tdrStyle.SetFuncColor(2)
  tdrStyle.SetFuncStyle(1)
  tdrStyle.SetFuncWidth(1)

  #For the date:
  tdrStyle.SetOptDate(0)
  #tdrStyle.SetDateX(Float_t x = 0.01)
  #tdrStyle.SetDateY(Float_t y = 0.01)

  #For the statistics box:
  tdrStyle.SetOptFile(0)
  tdrStyle.SetOptStat(0) #To display the mean and RMS:   SetOptStat('mr')
  tdrStyle.SetStatColor(rt.kWhite)
  tdrStyle.SetStatFont(42)
  tdrStyle.SetStatFontSize(0.025)
  tdrStyle.SetStatTextColor(1)
  tdrStyle.SetStatFormat('6.4g')
  tdrStyle.SetStatBorderSize(1)
  tdrStyle.SetStatH(0.1)
  tdrStyle.SetStatW(0.15)
  #tdrStyle.SetStatStyle(Style_t style = 1001)
  #tdrStyle.SetStatX(Float_t x = 0)
  #tdrStyle.SetStatY(Float_t y = 0)

  #Margins:
  tdrStyle.SetPadTopMargin(0.05)
  tdrStyle.SetPadBottomMargin(0.13)
  tdrStyle.SetPadLeftMargin(0.16)
  tdrStyle.SetPadRightMargin(0.02)

  #For the Global title:
  tdrStyle.SetOptTitle(0)
  tdrStyle.SetTitleFont(42)
  tdrStyle.SetTitleColor(1)
  tdrStyle.SetTitleTextColor(1)
  tdrStyle.SetTitleFillColor(10)
  tdrStyle.SetTitleFontSize(0.05)
  #tdrStyle.SetTitleH(0) #Set the height of the title box
  #tdrStyle.SetTitleW(0) #Set the width of the title box
  #tdrStyle.SetTitleX(0) #Set the position of the title box
  #tdrStyle.SetTitleY(0.985) #Set the position of the title box
  #tdrStyle.SetTitleStyle(Style_t style = 1001)
  #tdrStyle.SetTitleBorderSize(2)

  #For the axis titles:
  tdrStyle.SetTitleColor(1, 'XYZ')
  tdrStyle.SetTitleFont(42, 'XYZ')
  tdrStyle.SetTitleSize(0.06, 'XYZ')
  #tdrStyle.SetTitleXSize(Float_t size = 0.02) #Another way to set the size?
  #tdrStyle.SetTitleYSize(Float_t size = 0.02)
  tdrStyle.SetTitleXOffset(0.9)
  tdrStyle.SetTitleYOffset(1.25)
  #tdrStyle.SetTitleOffset(1.1, 'Y') #Another way to set the Offset

  #For the axis labels:
  tdrStyle.SetLabelColor(1, 'XYZ')
  tdrStyle.SetLabelFont(42, 'XYZ')
  tdrStyle.SetLabelOffset(0.007, 'XYZ')
  tdrStyle.SetLabelSize(0.05, 'XYZ')

  #For the axis:
  tdrStyle.SetAxisColor(1, 'XYZ')
  tdrStyle.SetStripDecimals(True)
  tdrStyle.SetTickLength(0.03, 'XYZ')
  #tdrStyle.SetNdivisions(510, 'XYZ')
  tdrStyle.SetNdivisions(505, 'XYZ')
  tdrStyle.SetPadTickX(1)  #To get tick marks on the opposite side of the frame
  tdrStyle.SetPadTickY(1)

  #Change for log plots:
  tdrStyle.SetOptLogx(0)
  tdrStyle.SetOptLogy(0)
  tdrStyle.SetOptLogz(0)

  #Legend font
  tdrStyle.SetLegendFont(42)

  #Postscript options:
  tdrStyle.SetPaperSize(20.,20.)
  #tdrStyle.SetLineScalePS(Float_t scale = 3)
  #tdrStyle.SetLineStyleString(Int_t i, const char* text)
  #tdrStyle.SetHeaderPS(const char* header)
  #tdrStyle.SetTitlePS(const char* pstitle)
  #
  #tdrStyle.SetBarOffset(Float_t baroff = 0.5)
  #tdrStyle.SetBarWidth(Float_t barwidth = 0.5)
  #tdrStyle.SetPaintTextFormat(const char* format = 'g')
  #tdrStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
  #tdrStyle.SetTimeOffset(Double_t toffset)
  #tdrStyle.SetHistMinimumZero(True)

  tdrStyle.SetHatchesLineWidth(5)
  tdrStyle.SetHatchesSpacing(0.05)

  tdrStyle.SetLineStyleString(11,'20 10')

  tdrStyle.cd()

def LDMX_lumi(pad, iPosX, extraText):
  #Global variables
  cmsText     = 'LDMX'
  cmsTextFont   = 52  #default is helvetic-italics

  writeExtraText = True
  extraTextFont = 52  #default is helvetica-italics
  
  #text sizes and text offsets with respect to the top frame
  #in unit of the top margin size
  lumiTextSize     = 0.6
  lumiTextOffset   = 0.2
  cmsTextSize      = 0.6
  cmsTextOffset    = 0.17 #0.1  #only used in outOfFrame version

  relPosX    = 0.3
  relPosY    = 0.035
  relExtraDY = 1.2

  #ratio of 'CMS' and extra text size
  extraOverCmsTextSize  = 0.76

  drawLogo      = False

  outOfFrame    = False
  if  iPosX/10==0 : 
      outOfFrame = True

  alignY_=3
  alignX_=2
  if  iPosX/10==0 : alignX_=1
  if  iPosX==0    : alignX_=1
  if  iPosX==0    : alignY_=1
  if  iPosX/10==1 : alignX_=1
  if  iPosX/10==2 : alignX_=2
  if  iPosX/10==3 : alignX_=3
  if  iPosX == 0  : relPosX = cmsTextOffset
  align_ = 10*alignX_ + alignY_

  H = pad.GetWh()
  W = pad.GetWw()
  l = pad.GetLeftMargin()
  t = pad.GetTopMargin()
  r = pad.GetRightMargin()
  b = pad.GetBottomMargin()
  # e = 0.025

  pad.cd()

  latex = rt.TLatex()
  latex.SetNDC()
  latex.SetTextAngle(0)
  latex.SetTextColor(rt.kBlack)    

  extraTextSize = extraOverCmsTextSize*cmsTextSize

  if  outOfFrame :
    latex.SetTextFont(cmsTextFont)
    latex.SetTextAlign(11) 
    latex.SetTextSize(cmsTextSize*t)    
    latex.DrawLatex(l + 0.55*(1-l-r),1-t+lumiTextOffset*t,cmsText)
  
  pad.cd()

  posX_=0
  if  iPosX%10<=1:
    posX_ =   l + relPosX*(1-l-r)
  elif  iPosX%10==2:
    posX_ =  l + 0.5*(1-l-r)
  elif  iPosX%10==3: 
    posX_ =  1-r - relPosX*(1-l-r)
  posY_ = 1-t - relPosY*(1-t-b)

  if not outOfFrame:
      latex.SetTextFont(cmsTextFont)
      latex.SetTextSize(cmsTextSize*t)
      latex.SetTextAlign(align_)
      latex.DrawLatex(posX_, posY_, cmsText)
      if writeExtraText:
        latex.SetTextFont(extraTextFont)
        latex.SetTextAlign(align_)
        latex.SetTextSize(extraTextSize*t)
        latex.DrawLatex(posX_, posY_- relExtraDY*cmsTextSize*t, extraText)
  elif writeExtraText:
    if iPosX==0:
      posX_ =   l +  (0.55+relPosX)*(1-l-r)
      posY_ =   1-t+lumiTextOffset*t
    latex.SetTextFont(extraTextFont)
    latex.SetTextSize(extraTextSize*t)
    latex.SetTextAlign(align_)
    latex.DrawLatex(posX_, posY_, extraText)      


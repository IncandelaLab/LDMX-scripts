import model ./ALP_Photophobic_UFO_Modified/
generate e- N > e- N p1, p1 > e- e+

output {{ dir_name }}

launch

shower=off
analysis=off
madspin=off
reweight=off
done

set mass 622 {{ mass_alp }}
set width 622 {{ width }}
set DMINPUTS lambda {{ lambda_e }}

set ebeam1 8.0
set ebeam2 171.3

set nevents {{ n_events }}
set iseed {{ seed }}

set ptl 0.0
set etal -1.0
set drll 0.0
done

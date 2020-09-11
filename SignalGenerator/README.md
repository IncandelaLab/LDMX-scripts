# Signal Setup

## Unpack files
```
tar -zxvf Events_fromNatalia.tar.gz
tar -zxvf ldmx_vector_decayed_Ebeam_4_GeV_rescaled_madgraph.tar.gz
```

## Compile and setup
```
cd ldmx_vector_decayed_Ebeam_4_GeV_rescaled/Source
make
```

Parameter files can be altered to give different mass spectrum for the Aprime and dark matter candidate. Current beam setup is for an ee beam with 10 times the energy and cross section due to some error in internal madgraph inputs. The files that need to be changed for different signals is Cards/param_card.dat for the particle masses and Cards/run_card.dat for the number of events (don't go over 100k events). 


## Running the MadGraph generator
Run with default setting and generate events
```
cd <work-dir>/SignalGenerator/ldmx_vector_decayed_Ebeam_4_GeV_rescaled
./bin/generate_events -f
```
To run with adjusted setting just leave out the -f


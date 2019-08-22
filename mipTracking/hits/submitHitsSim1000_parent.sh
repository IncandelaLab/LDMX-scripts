#1.0
bsub -R rhel60 -W 20 python /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits/sim1000MeV_parent.py --outdir /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits_copy_trees --outfile 1.0_tree_0.root --filelist /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits_copy_txts/filelist_1.0_3.txt --signal

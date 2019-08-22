#bkg
bsub -R rhel60 -W 20 python /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits/simbkg_parent.py --outdir /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits/hits_copy_trees --outfile bkg_tree_0.root --filelist /nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hits/hits_copy_txts/filelist_bkg_2.txt

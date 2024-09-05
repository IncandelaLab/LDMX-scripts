# Links to the main documentation for getting started
Generarly read through https://ldmx-software.github.io/

If you are not going to touch any of the .cpp files in ldmx-sw, you can use `denv` and have the software as an container image only
https://ldmx-software.github.io/using/getting-started.html#install-denv
Look up the possible tags here
https://github.com/LDMX-Software/ldmx-sw/tags

If you will change the .cpp files, use `just` as explained here
https://ldmx-software.github.io/developing/getting-started.html#install-just

# Running on ROOT files containing events

There is two way to do this:
1) The traditional way of looping through events and objects. 
An example config can be found here [runEventLoop](https://github.com/IncandelaLab/LDMX-scripts/blob/master/GeneralTutorial/runEventLoop.py)
Run it with the following (in case of not needing to change ldmx-sw)
```
mkdir myProject
cd myProject
denv init ldmx/pro:v4.1.0
wget https://raw.githubusercontent.com/IncandelaLab/LDMX-scripts/master/GeneralTutorial/runEventLoop.py
denv python3 runEventLoop.py  
```
3) Putting objects into data frames and have vectorized oparations. This is faster but can be harder to conseptualize. 
We have several examples in this repo, but also look at
https://ldmx-software.github.io/using/analysis/python.html

# nuNet

nuNet is a convolutional neural network platform for reconstructing neutrino properties from radio emission produced in Askaryan events

## Folder structure

    .
    ├── ...
    ├── common                  # directory with tools, models, ...
    │   ├── bandpass_filters
    │   ├── models
    │   │   ├── runD[n]         #  
    │   │   ├── runE[n]         # saved models from training sessions
    │   │   └── runF[n]         #
    │   └── tools
    │       └── start_training  # tools for running training sessions
    │
    ├── training                # directory for training code
    │   └── runs                
    │       ├── direction       
    │       │   ├── archive     # directory for archived runs and bases
    │       │   ├── base        # directory for bases 
    │       │   └── runD[n]     # n = 1, 2, 3, ...
    │       ├── energy
    │       │   ├── archive     #      
    │       │   ├── base        # ──── || ────
    │       │   └── runE[n]     #      
    │       └── flavour
    │           ├── archive     #
    │           ├── base        # ──── || ────
    │           └── runF[n]     #
    └── ...


## Packages


## Association

The work in this repository is done by Sigfrid Stjärnholm, as a part in the work as a research assistant at Uppsala University with Christian Glaser as a supervisor.
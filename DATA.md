# Local data contract

No research data are bundled. Obtain access from the dataset owner; a publication DOI does not grant redistribution rights. `--data-root` is the root of the corresponding original project, not this repository.

`paper/code_inpaper/healthy.mat`: MATLAB arrays `AME_dis_mat` and `current_mat`, observed shape (1000, 1801), one simulation per row. Select 90 points using linspace(0,1800,91)[1:]. Current and position must be aligned; time step after selection is 0.02 s.

## Preprocessing and split

Deterministic 80/20 split of complete simulations, followed by training-only position/current scaling and a median-distance physical envelope. Both training and held-out reconstruction run on CPU.

Missing files produce an actionable error. Dataset schemas were inspected locally; the synthetic demo only checks the code path and is not a substitute for research data.

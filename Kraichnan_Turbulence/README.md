# Kraichnan Turbulence:

## CNN training
- Run kraichnan_turbulence_dns.py file in './CNN/run_1', './CNN/run_2', './CNN/run_3', './CNN/run_4' folders.
- Run apriori_data.py file in './CNN/run_1', './CNN/run_2', './CNN/run_3', './CNN/run_4' folders.
- This will generate the training and test data for the CNN
- Run DHIT_CNN_apriori_sgs_v2.py file to train the CNN. The trained model and training statistics will be saved in 'nn_history' folder

## Data assimilation
- Run kraichnan_turbulence_enkf_v1 file in './DA/' folder. (You can use the 'w_fdns_8000.0_512_64.npz' file from any of the run_{} folder from the first step of the CNN training)

Unveiling the Distinctive Brain Functional Dynamics between Parkinson's Disease and Progressive Supranuclear Palsy 
====
This repository contains code in support of the paper,"Unveiling the Distinctive Brain Functional Dynamics between Parkinson's Disease and Progressive Supranuclear Palsy".

code
-----
The hmmleida folder contains the scripts required to conduct the main analyses:<br>
* run.py infers state switching of the participants based on the hmm model. input: preprocessed Bold time series. Output: state fractional occupancy, dwell time, transition probabilities and state time series for each subject.
* analysis.py performs further analyses on brain states identified by hmm analysis.Input: state fractional occupancy, dwell time, transition probabilities and state time series for each subject. Output: cluster centroids for each state, alignment results with the Yeo atlas, and comparisons of temporal biomarkers across different states.

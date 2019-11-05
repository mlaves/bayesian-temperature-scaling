# Well-calibrated Model Uncertainty with Temperature Scaling for Dropout Variational Inference

Authors:  
Max-Heinrich Laves, Sontje Ihler, Karl-Philipp Kortmann, Tobias Ortmaier

To appear at 4th workshop on Bayesian Deep Learning, Neural Information Processing Systems (NeurIPS)  2019.

[https://arxiv.org/abs/1909.13550](https://arxiv.org/abs/1909.13550])

## Abstract

Model uncertainty obtained by variational Bayesian inference with Monte Carlo dropout is prone to miscalibration.
The uncertainty does not represent the model error well.
In this paper, temperature scaling is extended to dropout variational inference to calibrate model uncertainty.
Expected uncertainty calibration error (UCE) is presented as a metric to measure miscalibration of uncertainty.
The effectiveness of this approach is evaluated on CIFAR-10/100 for recent CNN architectures.
After calibration, classification error is decreased by rejecting data samples with high uncertainty.
Experimental results show, that temperature scaling considerably reduces miscalibration by means of UCE and enables robust rejection of uncertain predictions.
The proposed approach can easily be derived from frequentist temperature scaling and yields well-calibrated model uncertainty.
It is simple to implement and does not affect the model accuracy.

## Contact

Max-Heinrich Laves  
[laves@imes.uni-hannover.de](mailto:laves@imes.uni-hannover.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Appelstr. 11A, 30167 Hannover, Germany

# On the Identifiability of Quantized Factors

This is the official reporitory for the code associated with the paper:
[On the Identifiability of Quantized Factors](https://arxiv.org/abs/2306.16334) by **Vitória Barin-Pacela, Kartik Ahuja, Simon Lacoste-Julien, Pascal Vincent**, Conference on Causal Learning Reasoning (CLeaR), 2024.

It contains notebooks and code for reproducing the figures in the paper.

- Figure 2 is reproduced in the notebook `exoplanet_data.ipynb`, it contains evidence of axis-aligned discontinuities in the Nasa Expolanets dataset.
-  Figure 3 is reproduced in the notebook `results_unfactorized.ipynb`, which contains a synthetic dataset with non-factorized support. We provide results for our model (axis alignment), Hausdorff Factorized Support, and Linear ICA.
- Figure 4 is reproduced in the notebook `results_factorized.ipynb`, which contains a synthetic dataset with factorized support. We provide results for our model (axis alignment) and Linear ICA.
- Figure 5 is reproduced in the notebook `mocap_data.ipynb`, it contains evidence of axis-aligned discontinuities in the CMU motion capture dataset.

<img width="550" src=main_fig.png>

## Requirements
Use `requirements.txt`.

## License
The majority of `quantized_identifiability` is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [`iVAE`](https://github.com/ilkhem/iVAE/tree/master) is licensed under the MIT license.


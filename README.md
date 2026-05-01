## Description

**Transformer-based model for inverse design of infrared (IR) optical filters.**
**Given target spectral characteristics - Reflectance (R), Transmittance (T), Dip, and Figure of Merit (FOM) the model predicts multilayer thin-film structures.**

### Overview

This project implements a sequence-generation approach to optical design using a custom Transformer architecture. The model maps continuous spectral targets to discrete layer sequences (material + thickness tokens).

Input: [R, T, Dip, FOM, Substrate]
Output: Ordered multilayer structure (e.g., Ag_47, TiO2_4, ...)

Core idea: treat thin-film design as a language generation problem.

### Future work

- Improve dataset
- Implement physics aware loss(TMM based)

Parameters at a single wavelength are not deterministic enough for an inverse design problem, hence evaluation of generated structures with TMM is crucial.

### Acknowledgements

This project is inspired by the OptoGPT framework for optical inverse design. It was an attempt at solving a narrower problem with a much smaller dataset and lesser compute.

Find the original paper here - https://arxiv.org/abs/2304.10294
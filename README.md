# 🥧 PIED: Physics-Informed Experimental Design for Inverse Problems

The code for paper titled "PIED: Physics-Informed Experimental Design for Inverse Problems". The paper has been accepted at ICLR 2025 (see https://openreview.net/forum?id=w7P92BEsb2).

> ### Abstract
> In many science and engineering settings, system dynamics are characterized by governing partial differential equations (PDEs), and a major challenge is to solve inverse problems (IPs) where unknown PDE parameters are inferred based on observational data gathered under limited budget. Due to the high costs of setting up and running experiments, experimental design (ED) is often done with the help of PDE simulations to optimize for the most informative design parameters (e.g., sensor placements) to solve such IPs, prior to actual data collection. This process of optimizing design parameters is especially critical when the budget and other practical constraints make it infeasible to adjust the design parameters between trials during the experiments. However, existing experimental design (ED) methods tend to require sequential and frequent design parameter adjustments between trials. Furthermore, they also have significant computational bottlenecks due to the need for complex numerical simulations for PDEs, and do not exploit the advantages provided by physics informed neural networks (PINNs) in solving IPs for PDE-governed systems, such as its meshless solutions, differentiability, and amortized training. This work presents Physics-Informed Experimental Design (PIED), the first ED framework that makes use of PINNs in a fully differentiable architecture to perform continuous optimization of design parameters for IPs for one-shot deployments. PIED overcomes existing methods' computational bottlenecks through parallelized computation and meta-learning of PINN parameter initialization, and proposes novel methods to effectively take into account PINN training dynamics in optimizing the ED parameters. Through experiments based on noisy simulated data and even real world experimental data, we empirically show that given limited observation budget, PIED significantly outperforms existing ED methods in solving IPs, including for challenging settings where the inverse parameters are unknown functions rather than just finite-dimensional.

## Setup

Run `pip install -r requirements.txt` to install the relevant packages. 

## Code Structure

- `pied/dataset` consists of all dataset used, not including those that are generated on the fly.

- `pied/test_cases.py` consists of all the test cases, returned as a dictionary. The dictionary would consist of the setup for the ED problem, along with an oracle that can be used to query the observations, and the true inverse parameter for that oracle.

- `pied/ed` consists of our ED method and other ED methods tested.

## Running Experiments

To run experiments, use `test_ed_loop.py` file. For example,
```
python test_ed_loop.py --problem osc --seed 0 --out-folder ../results-ed_loop
```
will run the damped oscillator (`osc`) dataset with seed 0 on all ED methods, and save the results to `../results-ed_loop`. See `test_ed_loop.py` to see all of the available datasets.

## Citation

```
@inproceedings{pied,
  title={{PIED: Physics-Informed Experimental Design for Inverse Problems}},
  author={Hemachandra, Apivich and Lau, Gregory Kang Ruey and Ng, See-Kiong and Low, Bryan Kian Hsiang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

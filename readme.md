## Self-Rewarding SMC for Masked Diffusion Language Models<br><sub>Official PyTorch Implementation</sub>
Self-Rewarding Sequential Monte Carlo (SMC) for Masked Diffusion Language Models

![sr-smc](images/overview.jpg)

## Overview
**TL;DR.** Self-Rewarding SMC is an inference-time scaling method that leverages trajectory-level confidence from diffusion models as importance weights to steer generation toward globally confident, high-quality samples.

- Self-Rewarding SMC is **reward-free** and thus can be applied to arbitrary pretrained models and tasks!

In this repository, we provide evaluations on standard Masked Diffusion Language Models (**MDLMs**) including MDLM and BD3-LMs, and diffusion large language models (**dLLMs**) including LLaDA-1.5 and Dream-7B.

### Code Structure

```text
.
├── 🚀bd3lms/          		   # Experiments on MDLM and BD3-LMs
│   ├── configs/		 		# Model and inference configurations
│	     └── config.yaml   		# main configurations, updated for [SMC]
│   ├── scripts/       	   		# sampling scripts 
│   └── diffusion.py     		# core algorithm, updated for [SMC]
│
├── 🚀llada/           		   # Experiments on LLaDA-1.5
│   ├── eval_llada.py			# Main evaluation code for LLaDA-1.5
│   ├── generate_smc.py			# Self-rewarding [SMC] implementation
│   └── eval.md         		# Evaluation instructions for LLaDA
│
├── 🚀dream/           		   # Experiments on Dream-7B
│   ├── model/
│		 └── generation_utils_smc_block.py 
								# Self-rewarding [SMC] implementation
│   ├── eval.py					# Main evaluation code for Dream-7B
│   └── eval.md					# Evaluation instructions for Dream
│
├── images/          			# Figures used in README
│
├── README.md        			# Project overview and instructions
└── LICENSE
```

Each subdirectory contains model-specific configurations and scripts for running self-rewarding SMC at inference time.

As an example, run the following code for sample quality evaluation:

```
cd bd3lms
sh scripts/gen_ppl/genppls_batch.sh
```
---

### Evaluation on MDLMs

**Self-rewarding SMC improves the generative perplexity:**

![sr-smc-mdlm](images/results-ppl.png)


### Evaluation on dLLMs

**1. Self-rewarding SMC improves dLLMs in math and coding:**

![sr-smc-dllms-1](images/tab-dllms.png)

**2. Overall performance trends as #particle increases:**
![sr-smc-dllms-2](images/n_particles.jpg)

**3. Effect of Gumbel noise temperature on model performance:**
![sr-smc-dllms-3](images/temperature.jpg)

### Contributing
Issues and Pull Requests are welcome!

### Acknowledgements
We would like to thank the authors of [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) and [bd3lms](https://github.com/kuleshov-group/bd3lms/tree/main) for their excellent work and open-source contributions.

### Citation

If you find this work useful, please cite our paper:

```
@article{luo2026self,
  title={Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models},
  author={Luo, Ziwei and Jin, Ziqi and Lei, Wang and Bing, Lidong and Sch{\"o}n, Thomas B},
  journal={arXiv preprint arXiv:2506.xxxx},
  year={2026}
}
```



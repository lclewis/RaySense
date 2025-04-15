# RaySense
V0.1 codes for the paper "Nearest Neighbor Sampling of Point Sets Using Rays"

### Installation:
the required packages are inside the env.yml file. Use conda or mamba (recommended) to install

`conda env create -f env.yml`

### Brief Summary:
RaySense is an novel and efficient tool to sample statistics and extract geometric information from the underlying geometric object based on closest point projection. See our paper: [Springer Nature](https://link.springer.com/article/10.1007/s42967-023-00318-1), [ArXiv](https://arxiv.org/pdf/1911.10737) for details.

See the example notebook for some preliminary applications, more to come.

### Reference:
Please considering citing our paper if you find it useful:

	@article{liu2024nearest,
  		title={Nearest neighbor sampling of point sets 	using rays},
  		author={Liu, Liangchen and Ly, Louis and Macdonald, Colin B and Tsai, Richard},
  		journal={Communications on Applied Mathematics and Computation},
  		volume={6},
  		number={2},
  		pages={1131--1174},
  		year={2024},
  		publisher={Springer}}

We use the CPU version of [faiss](https://faiss.ai) for efficient nearest neighbor search. A GPU version is also available if further acceleration is needed.


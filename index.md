@def title = "Xuanzhao Gao's Site"
@def tags = ["syntax", "code"]

# Greetings!

I am Xuanzhao Gao, a research fellow at the Flatiron Institute, Center for Computational Mathematics.
This is my blog website, hope you enjoy!
If you want to know more about me, please visit my [personal website](https://users.flatironinstitute.org/~xgao1/).

# Blogs

1. [How to implement generic matrix multiplication (GEMM) with generic element types on GPU?](/blogs/CuTropicalGEMM/)

    This blog is a technical note for the Open Source Promotion Plan 2023 project ["TropicalGEMM on GPU"](https://summer-ospp.ac.cn/org/prodetail/23fec0105?lang=en&list=pro) released by JuliaCN, where I developed a [Julia](https://julialang.org/) package [CuTropicalGemm.jl](https://github.com/TensorBFS/CuTropicalGEMM.jl) calculate Generic Matrix Multiplication (GEMM) of Tropical Numbers on Nvidia GPUs.

2. [Tensor Network Contraction Order Optimization with Exact Tree Width Solver](/blogs/contractionorder/)

    This blog is a technical note for the [Google Summer of Code 2024](https://summerofcode.withgoogle.com) project ["Tensor network contraction order optimization and visualization"](https://summerofcode.withgoogle.com/programs/2024/projects/B8qSy9dO) released by [The Julia Language](https://julialang.org/), where I implemented an optimizer for tensor network contraction order based on tree decomposition in the Julia package [OMEinsumContracionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).

3. [Finding the Optimal Tree Decomposition with Minimal Treewidth](/blogs/treewidth/)

    This blog is a supplementary for the note [Tensor Network Contraction Order Optimization with Exact Tree Width Solver](/blogs/contractionorder/), where I detailed introduce the algorithm to find the optimal tree decomposition with minimal treewidth of a given simple graph, and how it is implemented in Julia package [TreeWidthSolver.jl](https://github.com/xuanzhaogao/TreeWidthSolver.jl).

# Technical Notes

* [How to install slurm on Ubuntu 22.04](/blogs/slurm/)

  This blog is a technical note for the installation of slurm on Ubuntu 22.04 with NIS and apt tools.
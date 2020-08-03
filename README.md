# ConvOptDL

- Download the [data](https://drive.google.com/file/d/1Vd7br_DJGlaUEbKL0xLNXC0tAx9VuP2W/view?usp=sharing)

## Training

```bash
juliap train.jl --batches-per-episode=<INTEGER> --batch-size=<INTEGER> --num-episodes=<INTEGER> --log-period=<INTEGER> <DATA_FILE.jls>
```

ex: `juliap train.jl --batches-per-episode=400 --batch-size=20 --num-episodes=20 --log-period=20 CIFAR_FS-train.jls`

# TODOs

- GPU support
- implement `Block LU factorization` in the website below for better performance
  - https://locuslab.github.io/qpth/
- better QP solver: primal-dual interior point method 
  - As mentioned in [Mattingley & Boyd (2012)](https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf)

# References

- [Batched Kronecker product for 2-D matrices and 3-D arrays on NVIDIA GPUs](https://arxiv.org/pdf/1304.7054.pdf)
- [MetaOptNet](https://github.com/kjunelee/MetaOptNet)
- [qpth](https://github.com/locuslab/qpth)

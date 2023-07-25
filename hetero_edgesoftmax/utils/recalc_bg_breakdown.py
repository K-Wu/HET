from typing import Union


def recalc_best_from_baselines(
    all_baseline_csv: "list[list[Union[float, str, int]]]",
    model: str,
    in_dim: int,
    out_dim: int,
    num_heads: int,
):
    """
    An example output:
                          Inference
              aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    DGLa      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    DGLb      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    PyG       x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    Graphiler x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BEST      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
                          Training
              aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    DGLa      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    DGLb      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    PyG       x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    Seastar   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BEST      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    """
    pass


def recalc_best_of_hector(
    all_hector_csv: "list[list[Union[float, str, int]]]",
    model: str,
    in_dim: int,
    out_dim: int,
    num_heads: int,
):
    """
    An example output:
                          Inference
        aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    U   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    F   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C+F x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BST x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
                          Training
        aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    U   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    F   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C+F x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BST x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    """
    pass


def recalc_worst_mean_best(
    all_hector_csv: "list[list[Union[float, str, int]]]",
    all_baseline_csv: "list[list[Union[float, str, int]]]",
    model: str,
    in_dim: int,
    out_dim: int,
    num_heads: int,
):
    """
    An example output:
          Training					Inference
      #degradation	worst	mean	best	#oom by competitors	#degradation	worst	mean	best	#oom by competitors
                              unoptimized
        RGCN	1.00	0.93	1.80	4.59	2.00	1.00	0.97	1.44	3.74	0.00
      RGAT	0.00	4.36	4.93	5.59	6.00	0.00	5.31	6.39	7.76	2.00
      HGT	  1.00	1.66	1.88	0.98	2.00	0.00	0.77	1.19	1.98	2.00
                              most optimized
      RGCN	1.00	0.93	1.80	4.59	2.00	1.00	0.97	1.44	3.74	0.00
      RGAT	0.00	4.36	4.93	5.59	6.00	0.00	5.31	6.39	7.76	2.00
      HGT	  1.00	1.66	1.88	0.98	2.00	0.00	0.77	1.19	1.98	2.00
    """
    pass


def recalc_opt_matrix(
    all_hector_csv: "list[list[Union[float, str, int]]]",
    model: str,
    in_dim: int,
    out_dim: int,
    num_heads: int,
):
    """
    An example output:
        Training Opt.			Inference Opt.
        C	F	C+F	C	F	C+F
                    RGAT
        aifb	0.84	1.17	0.85	1.04	1.24	1.12
      mutag	0.76	1.14	0.80	1.16	1.25	1.33
      bgs	0.94	1.20	1.06	1.17	1.38	1.41
      am	0.85	1.15	0.93	0.94	1.34	1.08
      mag	1*	OOM	1.02	1*	OOM	1.02
      wikikg2	1*	OOM	1.08	1*	OOM	1.03
      fb15k	1.29	1.25	1.40	1.58	1.40	1.79
      biokg	2.16	1.26	2.20	1.93	1.41	1.97
      AVERAGE	1.06	1.19	1.13	1.26	1.33	1.41
                    HGT
        aifb	1.88	1.26	1.86	1.80	1.82	1.61
      mutag	1.28	1.16	1.32	1.46	1.54	1.36
      bgs	1.20	1.13	1.19	1.27	1.26	1.07
      am	1.11	1.10	1.07	1.29	1.30	0.93
      mag	1.08	1.08	1.14	1.11	1.11	0.90
      wikikg2	1.10	1.09	1.20	1.13	1.13	0.99
      fb15k	1.24	1.14	1.31	1.29	1.29	1.20
      biokg	1.04	1.02	1.15	1.05	1.05	0.95
      AVERAGE	1.22	1.12	1.26	1.28	1.29	1.11
    """
    pass

Use the following scripting to Set up LD_LIBRARY_PATH as graphiler's dependency requires nvrtc 11.1
```
conda activate graphiler
conda env config vars set LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
```

Source: https://stackoverflow.com/questions/31598963/how-to-set-specific-environment-variables-when-activating-conda-environment
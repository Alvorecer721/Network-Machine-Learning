# EE452 - Networks machine learning project

## Project structure
```bash
.
├── data/  # Original dataset unzipped, per city.
├── exploitation
│   ├── clustering_city.ipynb  # Notebook with actual training calls and plotting.
│   ├── data/  # Numpy arrays containing the pre-processed features.
│   ├── models.py  # Definition of GNNs.
│   ├── prepare_data.py  # Prepare data for the actual training.
│   ├── training.ipynb  # Examples of how to use the training loop APIs.
│   └── training.py  # Training & evaluation loops for classical and PyG models.
├── exploration_final
│   ├── exploration_final.ipynb  # Main exploration; print statistics and make plots.
│   ├── spectral/  # Caching directory to save Laplacian's eigendecomposition for each city.
│   └── statistics/  # Caching directory for simple statistics of each city.
├── requirements.txt  # Libraries used throughout the project.
└── utils.py  # Project's utils, most importantly to construct graphs from the original data.

```

## Reproducing the results
1. Begin by creating a conda environment and install the requirements.

2. Download the original data from https://zenodo.org/record/1186215. You should unzip every city's folder inside of the top-level `data` directory (see above).

3. The exploitation part requires the eigen-decomposition of the graph Laplacian, which is computed by the exploration part. Therefore, you should run the `exploration_final.ipynb` notebook next.

4. Finally, for the exploitation part, you should start by generating the features. To do so, copy the `spectral` directory created by `exploration_final.ipynb` into `exploitation/data` and run `exploitation/prepare_data.py` in an environment with libraries from `requirements.txt` installed.

5. You are now ready to train our models, which you can do by running the notebook `exploitation/clustering_city.ipynb`.
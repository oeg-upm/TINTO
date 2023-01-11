**TINTO** is novel algorithm for converting **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** into synthetic images ...

## Main Features

- Supports all CSV data in **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** format.
- For now, the algorithm converts tabular data for binary and multi-class classification problems into machine learning.
- Input data formats:
    - **Tabular files**: The input data must be in **[CSV](https://en.wikipedia.org/wiki/Comma-separated_values)**, taking into account the **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** format.
    - **Tidy Data**: The **target** (variable to be predicted) should be set as the last column of the dataset. Therefore, the first columns will be the features.
- Two dimensionality reduction algorithms are used in image creation, **[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)** and **[*t*-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)** from the Scikit-learn Python library.
- The synthetic images to be created will be in black and white, i.e. in 1 channel.
- The synthetic image **dimensions** can be set as a parameter when creating them.
- The synthetic images can be created using **characteristic pixels** or **blurring** painting technique (expressing an overlap of pixels as the **maximum** or **average**).
- Runs on **Linux**, **Windows** and **macOS** systems.
- Compatible with **[Python](https://www.python.org/)** 3.7 or higher.

## Citing

If you used TINTO in your work, please cite the **[Information Fusion Journal](https://doi.org/10.1016/j.inffus.2022.10.011)**:

```bib
@article{inffus,
  title={A novel deep learning approach using blurring image techniques for Bluetooth-based indoor localisation},
  author={Talla-Chumpitaz, Reewos and Castillo-Cara, Manuel and Orozco-Barbosa, Luis and Garc{\'\i}a-Castro, Ra{\'u}l},
  journal={Information Fusion},
  volume={91},
  pages={173--186},
  year={2023},
  publisher={Elsevier}
}
```

## Licenses

**TINTO** is available under the **[Apache License 2.0](https://github.com/manwestc/TINTO/blob/main/LICENSE)**.

The **documentation** is licensed under **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.

## Authors

- **[Manuel Castillo-Cara](https://github.com/manwestc) - [jcastillo@fi.upm.es](mailto:jcastillo@fi.upm.es)**
- **[Raúl García-Castro](https://github.com/rgcmme)**

*[Ontology Engineering Group](https://oeg.fi.upm.es)*, *[Universidad Politécnica de Madrid](https://www.upm.es/internacional)*.

## Contributors

See the full list of contributors **[here](https://github.com/manwestc/TINTO/graphs/contributors)**.

<img src="assets/logo-oeg.png" alt="Ontology Engineering Group" width="200"> <img src="assets/logo-upm.png" alt="Universidad Politécnica de Madrid" width="200"> <img src="assets/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="170">

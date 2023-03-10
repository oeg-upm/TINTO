[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/manwestc/TINTO/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7463973.svg)](https://doi.org/10.5281/zenodo.7463973)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)
[![Documentation Status](https://readthedocs.org/projects/morph-kgc/badge/?version=latest)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)

**TINTO** is an engine that constructs **Synthetic Images** from [Tidy Data](https://www.jstatsoft.org/article/view/v059i10) (also knows as **Tabular Data**). 

**Citing TINTO**: If you used TINTO in your work, please cite the **[INFFUS Paper](https://doi.org/10.1016/j.inffus.2022.10.011)**:

```bib
@article{inffus_TINTO,
    title = {A novel deep learning approach using blurring image techniques for Bluetooth-based indoor localisation},
    journal = {Information Fusion},
    author = {Reewos Talla-Chumpitaz and Manuel Castillo-Cara and Luis Orozco-Barbosa and Raúl García-Castro},
    volume = {91},
    pages = {173-186},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2022.10.011}
}
```

## Main Features

- Supports all CSV data in **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** format.
- For now, the algorithm converts tabular data for binary and multi-class classification problems into machine learning.
- Input data formats:
    - **Tabular files**: The input data must be in **[CSV](https://en.wikipedia.org/wiki/Comma-separated_values)**, taking into account the **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** format.
    - **Tidy Data**: The **target** (variable to be predicted) should be set as the last column of the dataset. Therefore, the first columns will be the features.
    - All data must be in numerical form. TINTO does not accept data in string or any other non-numeric format.
- Two dimensionality reduction algorithms are used in image creation, **[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)** and **[*t*-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)** from the Scikit-learn Python library.
- The synthetic images to be created will be in black and white, i.e. in 1 channel.
- The synthetic image **dimensions** can be set as a parameter when creating them.
- The synthetic images can be created using **characteristic pixels** or **blurring** painting technique (expressing an overlap of pixels as the **maximum** or **average**).
- Runs on **Linux**, **Windows** and **macOS** systems.
- Compatible with **[Python](https://www.python.org/)** 3.7 or higher.

## Video Documentation

https://user-images.githubusercontent.com/102165947/217485660-ca7e936a-e9bb-48a3-aaa4-3ba003bac36d.mp4


<!--- **[Read the documentation](https://readthedocs.io/en/latest/documentation/)**. -->

## Getting Started

**[TINTO](https://github.com/oeg-upm/TINTO)** is easy to use in terminal:

Fist, it is important to install all previus libraries
```
    pip install -r requirements.txt
```


To run the engine via **command line** and see all the **arguments** you just need to execute the following:
```
    python tinto.py -h
```

![Help](https://github.com/manwestc/TINTO/blob/main/imgs/tinto_help.png)

The default parameter are the following:
- **Dimensional Reduction Algorithm (-alg)**: Select the dimensionality reduction algorithm to be used for image creation. The [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)** or [*t*-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) algorithms can be chosen. By default, use the [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)** algorithm.
- **Image size (-px)**: 20x20 pixels
- **Blurring (-B)**: for default is False, i.e., it do not use Blurring technique and create de images with characteristic pixels
- **Amplification (-aB)**: Only if Blurring is True. It is the blurring amplification and for default is PI number, i.e., 3.141592653589793 aprox.
- **Blurring distance (-dB)**: Only if Blurring is True. It is Blurring distance and for default is 0.1 (10%).
- **Blurring steps (-sB)**: Only if Blurring is True. It is Blurring steps and for default is 4, i.e., expand 4 pixels the blurring.
- **Blurring option (-oB)**: Only if Blurring is True. It is the Blurring option and for default is _mean_, i.e., if two pixels are overlaping, calculate the average number of this two overlaping pixels.
- **Save Configuration (-sC)**: Save the configurarion in a pikle object. It is False for default.
- **Load Configuration (-lC)**: Load the configurarion in a pikle object. It is False for default.
- **Seed (-sd)**: Set a seed for the random numbers. It is 20 for default.
- **_t_SNE times replication (-tt)**: It is only used when _t_-SNE is used. It is _t_-SNE times replication and for defaultd is 4.
- **Verbose (-v)**. Show in terminal the execution. For default is False.

## Previous considerations
Please note that the following considerations must be taken into account before running the script:
- Data must be in CSV with the default separator, i.e., commas.
- Only create images when we have data for a binary or multi-class classification problem.
- The last column should be the targer (variable to predict).
- The first columns will be the characteristics.
- All variables must be in numerical format.
- The script takes by default the first row as the name of each feature, therefore, the different features must be named.
- Each sample (row) of the dataset will correspond to an image.

For example, the following table shows a classic example of the [IRIS CSV dataset](https://archive.ics.uci.edu/ml/datasets/iris) as it should look like for the run:

| sepal length | sepal width | petal length | petal width | target |
|--------------|-------------|--------------|-------------|--------|
| 4.9          | 3.0         | 1.4          | 0.2         | 1      |
| 7.0          | 3.2         | 4.7          | 1.4         | 2      |
| 6.3          | 3.3         | 6.0          | 2.5         | 3      |

## Simple example without Blurring
The following example shows how to create 20x20 images with characteristic pixels, i.e. without blurring. 

```
    python tinto.py "iris.csv" "iris_images"
```

The images are created with the following considerations regarding the parameters used:
- **python**: to launch the Python script
- **tinto.py**: the name of the script
- **iris.csv**: the dataset to use. In this example, the IRIS dataset is used.
- **iris/**: the folder where the images will be saved.

Also, as no other parameters are indicated, you will choose the following parameters which are set by default:
- **Image size**: 20x20 pixels
- **Blurring**: No blurring will be used.
- **Seed**: with the seed set to 20.

Within the folder named "iris/" we can find subfolders with numbers where each number corresponds to the target used. For example, for the dataset iris.csv we will have three subfolders named "1/", "2/" and "3/". The following Figure shows an image created according to the example seen.

<kbd> ![Characteristic](https://github.com/manwestc/TINTO/blob/main/imgs/characteristic.png) </kbd>


## More specific example
The following example shows how to create with blurring with a more especific parameters.

```
    python tinto.py "iris.csv" "iris_images_tSNE" -B -alg t-SNE -oB maximum -px 30 -sB 5
```

The images are created with the following considerations regarding the parameters used:
- **Blurring (-B)**: Create the images with blurring technique.
- **Dimensional Reduction Algorithm (-alg)**: t-SNE is used.
- **Blurring option (-oB)**: Create de images with maximum value of overlaping pixel
- **Image size (-px)**: 30x30 pixels
- **Blurring steps (-sB)**: Expand 5 pixels the blurring.

<kbd> ![Blurring](https://github.com/manwestc/TINTO/blob/main/imgs/blurring.png) </kbd>

## License

TINTO is available under the **[Apache License 2.0](https://github.com/manwestc/TINTO/blob/main/LICENSE)**.

## Authors

- **[Manuel Castillo-Cara](https://github.com/manwestc) - [jcastillo@fi.upm.es](mailto:jcastillo@fi.upm.es)**
- **[Raúl García-Castro](https://github.com/rgcmme)**

*[Ontology Engineering Group](https://oeg.fi.upm.es)*, *[Universidad Politécnica de Madrid](https://www.upm.es/internacional)*.

## Contributors

See the full list of contributors **[here](https://github.com/manwestc/TINTO/graphs/contributors)**.

<kbd><img src="assets/logo-oeg.png" alt="Ontology Engineering Group" width="200"></kbd> <kbd><img src="assets/logo-upm.png" alt="Universidad Politécnica de Madrid" width="200"></kbd> <kbd><img src="assets/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="155"></kbd> 

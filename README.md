# TINTO: Converting Tidy Data into Image for Classification with 2D CNNs

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/manwestc/TINTO/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7463973.svg)](https://doi.org/10.5281/zenodo.7463973)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1plFq1CpEXIdc9LankaLPiOObRg0_y5l2?usp=sharing)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oeg-upm/TINTO)

<p align="center">
  <img src="imgs/logo.svg" alt="TINTO Logo" width="150">
</p>

---

> ⚠️ **Important Notice**
>
> This repository contains the original implementation of **TINTO**, an engine for converting tabular (tidy) data into synthetic images using dimensionality reduction and convolution techniques.
>
> 📦 **We strongly recommend using the updated library [TINTOlib](https://github.com/oeg-upm/TINTOlib)**, which includes:
>
> - The original **TINTO** method
> - Several additional methods such as **IGTD**, **REFINED**, **BarGraph**, **DistanceMatrix**, **Combination**, **FeatureWrap**, **SuperTML**, and **BIE**
> - A much more user-friendly and flexible interface
> - Complete and regularly updated documentation
> - A free course with examples, notebooks, and video tutorials
>
> 🔄 **TINTOlib** is under active development and continues to receive improvements.
>
> 👉 **For new projects and applications, we highly recommend switching to [TINTOlib](https://github.com/oeg-upm/TINTOlib).**

---

## 🚀 Overview

**TINTO** is a Python engine to transform **Tidy Data** (aka **tabular data**) into **synthetic images**, enabling CNN-based classification on non-visual datasets.

---

## 🔍 Explore with DeepWiki

TINTO has a dedicated page in **[DeepWiki](https://deepwiki.com/oeg-upm/TINTO)**, where you can browse semantic documentation, use cases, FAQs, and more.

<p align="center">
  <a href="https://deepwiki.com/oeg-upm/TINTO" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" />
  </a>
</p>

---

## 🧠 Key Features

* 📊 Input: Tidy CSV files (target as last column)
* 🎯 Supports binary and multi-class classification
* 🔄 Dimensionality reduction: PCA and t-SNE
* 🖼️ Output: grayscale synthetic images with/without blurring
* ⚙️ Customizable image size and overlap strategies
* 🐍 Python 3.7+ compatible

---

## 📽️ Quick Demo

[https://user-images.githubusercontent.com/102165947/217485660-ca7e936a-e9bb-48a3-aaa4-3ba003bac36d.mp4](https://user-images.githubusercontent.com/102165947/217485660-ca7e936a-e9bb-48a3-aaa4-3ba003bac36d.mp4)

---

## 💾 Installation & Usage

```bash
pip install -r requirements.txt
python tinto.py -h  # view all available options
```

**Example - No Blurring:**

```bash
python tinto.py iris.csv iris_images
```

**Example - With Blurring + t-SNE:**

```bash
python tinto.py iris.csv iris_images_tSNE -B -alg t-SNE -oB maximum -px 30 -sB 5
```

---

## 📊 Example Dataset Format (Iris)

| sepal length | sepal width | petal length | petal width | target |
| ------------ | ----------- | ------------ | ----------- | ------ |
| 4.9          | 3.0         | 1.4          | 0.2         | 1      |
| 7.0          | 3.2         | 4.7          | 1.4         | 2      |
| 6.3          | 3.3         | 6.0          | 2.5         | 3      |

---

## 🧪 Output Examples

<p align="center">
  <kbd><img src="https://github.com/manwestc/TINTO/blob/main/imgs/characteristic.png" alt="TINTO pixel" width="250"></kbd>
  <kbd><img src="https://github.com/manwestc/TINTO/blob/main/imgs/blurring.png" alt="TINTO blurring" width="250"></kbd>
</p>

---

## 📚 Citation

If you use **TINTO**, please cite:

```bib
@article{softwarex_TINTO,
  title = {TINTO: Converting Tidy Data into Image for Classification with 2-Dimensional Convolutional Neural Networks},
  journal = {SoftwareX},
  author = {Manuel Castillo-Cara and Reewos Talla-Chumpitaz and Raúl García-Castro and Luis Orozco-Barbosa},
  volume = {22},
  pages = {101391},
  year = {2023},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2023.101391}
}
```

And for indoor localisation use-case:

```bib
@article{inffus_TINTO,
  title = {A novel deep learning approach using blurring image techniques for Bluetooth-based indoor localisation},
  journal = {Information Fusion},
  author = {Reewos Talla-Chumpitaz and Manuel Castillo-Cara and Luis Orozco-Barbosa and Raúl García-Castro},
  volume = {91},
  pages = {173-186},
  year = {2023},
  doi = {https://doi.org/10.1016/j.inffus.2022.10.011}
}
```

---

## 📘 Colab Tutorial

Learn how to load the generated images into CNNs:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1plFq1CpEXIdc9LankaLPiOObRg0_y5l2?usp=sharing)

---

## 👥 Authors & Contributors

* **[Manuel Castillo-Cara](https://github.com/manwestc)**
  [jcastillo@fi.upm.es](mailto:jcastillo@fi.upm.es)
* **[Raúl García-Castro](https://github.com/rgcmme)**

<p align="center">
  <kbd><img src="assets/logo-oeg.png" alt="OEG" width="150"></kbd>
  <kbd><img src="assets/logo-upm.png" alt="UPM" width="150"></kbd>
  <kbd><img src="assets/logo-uned-.jpg" alt="UNED" width="231"></kbd>
  <kbd><img src="assets/logo-uclm.png" alt="UCLM" width="115"></kbd>
</p>

---

## 🛡️ License

TINTO is released under the **[Apache License 2.0](https://github.com/manwestc/TINTO/blob/main/LICENSE)**.

# Improving Fairness in Medical Imaging Through Causal Learning

## Overview
![图片](https://github.com/ncbi-nlp/Medical-Imaging-Causal-Fairness/assets/1357144/19c11c0c-251e-4c8a-8b31-905a78f78c66)

This is the official repo for the article "Improving Fairness in Medical Imaging Through Causal Learning". 

In recent years, there has been significant attention towards the application of medical AI, which has demonstrated expert-level diagnosis capabilities and proven valuable in clinical settings. However, while considerable efforts have been devoted to improving model performance, there has been a limited focus on addressing the issue of AI fairness in biomedicine. This lack of attention to fairness can have potentially catastrophic consequences. Previous research has revealed that even state-of-the-art medical imaging models can exhibit unfair behavior and lower performance when it comes to minority demographic groups present in the dataset. In response, this study aims to address fairness in image-based computer-aided diagnosis. Specifically, we propose a universal causal fairness module that can be used across different backbone models in medical AI. In addition to the image feature, our proposed method learns the underlying causal relationships between clinical observation and sensitive attributes, following the intuition that real-world healthcare practitioners use not only clinical images but also consider the patient’s other attributes, like sex and age. To reduce bias, we add noise to the embedding of sensitive attributes, discouraging the classifier from exploiting skewed data distributions. We experiment with our method on three large medical imaging datasets and show significant fairness improvement with respect to all sensitive attributes while at no cost of overall performance.

## Code 
The python codes to reproduce our work, including training code, model architectures are in scripts. 

## Data availability
The Chexpert dataset is available at the website of the Center for Artificial Intelligence in Medicine & Imaging (https://aimi.stanford.edu/chexpert-chest-x-rays). The MIMIC-CXR dataset is available on PhysioNet (https://www.physionet.org/content/mimic-cxr-jpg/). The MIDRC dataset is available at the organization’s website (https://data.midrc.org/). 

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This work shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

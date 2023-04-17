# Towards Harnessing Feature Embedding for Robust Learning with Noisy Labels

This repository is the official implementation of the MLJ'22 paper Towards Harnessing Feature Embedding for Robust Learning with Noisy Label, authored by Chuang Zhang, Li Shen, Jian Yang, and Chen Gong. Our method utilizes the perperty that the embedded features induced by the memorization effect are more robust than the model-predicted labels, dealing with the classification problem under lable noise.

**Key Words**: Label Noise Learning, Robust Machine Learning

**Abstract**: The memorization effect of deep neural networks (DNNs) plays a pivotal role in recent label noise learning methods. To exploit this effect, the model prediction-based methods have been widely adopted, which aim to exploit the outputs of DNNs in the early stage of learning to correct noisy labels. However, we observe that the model will make mistakes during label prediction, resulting in unsatisfactory performance. By contrast, the produced features in the early stage of learning show better robustness. Inspired by this observation, in this paper, we propose a novel feature embedding-based method for deep learning with label noise, termed LabEl Noise Dilution (LEND). To be specific, we first compute a similarity matrix based on current embedded features to capture the local structure of training data. Then, the noisy supervision signals carried by mislabeled data are overwhelmed by nearby correctly labeled ones (i.e., label noise dilution), of which the effectiveness is guaranteed by the inherent robustness of feature embedding. Finally, the training data with diluted labels are further used to train a robust classifier. Empirically, we conduct extensive experiments on both synthetic and real-world noisy datasets by comparing our LEND with several representative robust learning approaches. The results verify the effectiveness of our LEND.


```
@article{zhang2022towards,
  title={Towards harnessing feature embedding for robust learning with noisy labels},
  author={Zhang, Chuang and Shen, Li and Yang, Jian and Gong, Chen},
  journal={Machine Learning},
  volume={111},
  number={9},
  pages={3181--3201},
  year={2022},
  publisher={Springer}
}

```


## Get Started

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

###  Datasets

Please download the datasets in folder

```
../data/
```

- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

- [Animal-10N](https://dm.kaist.ac.kr/datasets/animal-10n/)




## Training

To train the watermarking on CIFAR benckmarks, simply run:

- CIFAR-10
```train cifar10
python train.py cifar10 
```


- CIFAR-100
```train cifar100
python train.py cifar100
```


- Animal-10N
```train cifar100
python train.py animal10n
```
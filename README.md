![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/shesshan/CEIB/blob/main/LICENSE)
[![AAAI-24](https://img.shields.io/badge/AAAI-2024-black?color=%23FFA500)](https://aaai.org/main-track/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23FF6347.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# 🪄CEIB

Hi there👋, this repo contains the PyTorch implementation for our paper:

[Counterfactual-Enhanced Information Bottleneck for Aspect-Based Sentiment Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/29726/31247) 

Mingshan Chang, Min Yang, Qingshan Jiang, Ruifeng Xu. AAAI, 2024. 

## 📜 Summary
> Great success in the ABSA task? We found that deep ABSA models are prone to learning 🫧***spurious correlations***🫧 between input features and output labels, resulting in poor robustness and generalization capability!

📌 An example of the spurious correlation problem in ABSA:

<img src="/docs/example.png" width = "50%" />

To reduce **spurious correlations** for ABSA, we propose a novel **C**ounterfactual-**E**nhanced **I**nformation **B**ottleneck framework (called **CEIB**), which extends the information bottleneck (IB) principle to a factual-counterfactual balancing setting and incorporates augmented counterfactual data, to learn more robust and generalizable representations.
- We employ the IB principle to discard spurious features or shallow patterns while retaining sufficient information about the sentiment label.
- We devise a multi-pattern prompting method, which utilizes LLM to automatically generate counterfactual samples featuring identical spurious context words but opposite sentiment polarities for the original training data.
- We separate the mutual information in the original IB objective into factual and counterfactual parts. By balancing the predictive information of these two parts, we can learn more robust and generalizable representations against the dataset bias.

## 🧩 Architecture
<img src="/docs/CEIB_framework.png" width = "90%" />

(a) a **counterfactual data augmentation** module that utilizes LLM to generate counterfactual data for the original training data.

(b) an **information bottleneck** module with a factual-counterfactual balancing setting to learn more robust and generalizable representations.

## 🎯 Main Results
<img src="/docs/main_results.png" width = "90%" />

## 🗂 Code & Data
### Requirements
- Python 3.9.7
- PyTorch 1.11.0
- [Transformers](https://github.com/huggingface/transformers) 4.18.0
- CUDA 11.0

### Preparation
-  **Data** <br>
We have provided the generated counterfactual data in [data/augmented_t5_xxl/](/data/augmented_t5_xxl/).
<br> You may also run the command: `bash aug_data.sh` to generate the counterfactual data yourself. Before that, you should download [t5-xxl](https://huggingface.co/t5-11b) and set the parameter `--model_name_or_path` in [aug_data.sh](/aug_data.sh) to your local directory.

-  **Models** <br>
Download the PyTorch version [bert-base-uncased](https://huggingface.co/bert-base-uncased) and set the parameter `--model_dir` to your local directory.

 
### Training
- Run the command: `bash run_CEIB_xxx.sh`, e.g. run `bash run_CEIB_res14.sh` to train with REST14 dataset.

- More arguments can be found in [run.py](/run.py). Feel free to set parameters e.g. `--save_folder`(to save training results) and `--data_dir`(to load training&testing data) to your customized path.

### Citation
```bibtex
@article{Chang_Yang_Jiang_Xu_2024,
  title={Counterfactual-Enhanced Information Bottleneck for Aspect-Based Sentiment Analysis},
  volume={38},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/29726}, DOI={10.1609/aaai.v38i16.29726},
  number={16},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Chang, Mingshan and Yang, Min and Jiang, Qingshan and Xu, Ruifeng},
  year={2024},
  month={Mar.},
  pages={17736-17744}
}
```

🤘Please cite our paper and kindly give a star if you find this repo helpful.

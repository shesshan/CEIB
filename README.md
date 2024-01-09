![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/shesshan/CEIB/blob/main/LICENSE)
[![AAAI-24](https://img.shields.io/badge/AAAI_2024-Paper_11284-black?labelColor=%233CB371&color=%23FFA500)](https://aaai.org/main-track/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23FF6347.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# CEIB

Hi there👋! I'm delighted to share our paper [Counterfactual-Enhanced Information Bottleneck for Aspect-Based Sentiment Analysis](https://drive.google.com/file/d/1T3gJ_Dp67Buw7bR-p3ndFTJ1A1gmtILg/view?usp=drive_link) presents in AAAI 2024.

<!-- [Counterfactual-Enhanced Information Bottleneck for Aspect-Based Sentiment Analysis](https://drive.google.com/file/d/1T3gJ_Dp67Buw7bR-p3ndFTJ1A1gmtILg/view?usp=drive_link) <br> Mingshan Chang, Min Yang<sup>*</sup>, Qingshan Jiang, Ruifeng Xu. AAAI, 2024.--> 

>🔎 Huge success in the ABSA task❓ 🧐 We found that deep neural networks are prone to learning ***spurious correlations*** between input features and output labels, leading to poor robustness and generalization capability❗️

<img src="/docs/example.png" width = "55%" />

In this paper, we propose a novel Counterfactual-Enhanced Information Bottleneck framework (called **CEIB**) to mitigate the spurious correlation problem for ABSA. Concretely, **(1)** we employed the information bottleneck (IB) principle to discard superfluous information and shallow patterns while preserving sufficient information about the sentiment labels; **(2)** extended the original IB to a factual-counterfactual balancing setting to learn more robust and balanced representations against the dataset bias; **(3)** devised a multi-pattern prompting method and leveraged LLM to automatically generate counterfactual samples with promising quality.

## 🧩 Model Architecture
<img src="/docs/CEIB_framework.png" width = "100%" />

## 🗂 Code & Data

# REDRESS
Open-source code for ''Individual Fairness for Graph Neural Networks: A Ranking based Approach''.

## Citation

If you find it useful, please cite our paper. Thank you!

@inproceedings{dong2021individual,
  title={Individual fairness for graph neural networks: A ranking based approach},
  author={Dong, Yushun and Kang, Jian and Tong, Hanghang and Li, Jundong},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={300--310},
  year={2021}
}

## Environment
Experiments are carried out on a Titan RTX with Cuda 10.1. 

Details can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
Default dataset for node classification and link prediction is ACM and BlogCatalog, respectively.
Use as
```
python xxx_NDCG.py
```
or
```
python xxx_ERR.py
```

## Log example for node classification

```
python REDRESS_feature_NDCG.py
```
Log example:
```
ACM
Using ACM dataset
Epoch: 0001 loss_train: 2.1885 acc_train: 0.1177 loss_val: 2.1724 acc_val: 0.1675 time: 0.3276s
Epoch: 0002 loss_train: 2.1617 acc_train: 0.2257 loss_val: 2.1533 acc_val: 0.2700 time: 0.0070s
Epoch: 0003 loss_train: 2.1352 acc_train: 0.3580 loss_val: 2.1344 acc_val: 0.3617 time: 0.0070s
Epoch: 0004 loss_train: 2.1089 acc_train: 0.4903 loss_val: 2.1158 acc_val: 0.4157 time: 0.0050s
Epoch: 0005 loss_train: 2.0830 acc_train: 0.5752 loss_val: 2.0974 acc_val: 0.4515 time: 0.0050s
Epoch: 0006 loss_train: 2.0575 acc_train: 0.6056 loss_val: 2.0792 acc_val: 0.4794 time: 0.0040s
Epoch: 0007 loss_train: 2.0322 acc_train: 0.6201 loss_val: 2.0613 acc_val: 0.4988 time: 0.0040s
Epoch: 0008 loss_train: 2.0073 acc_train: 0.6359 loss_val: 2.0437 acc_val: 0.5109 time: 0.0040s
Epoch: 0009 loss_train: 1.9828 acc_train: 0.6468 loss_val: 2.0264 acc_val: 0.5261 time: 0.0030s
Epoch: 0010 loss_train: 1.9586 acc_train: 0.6541 loss_val: 2.0094 acc_val: 0.5322 time: 0.0040s
Epoch: 0011 loss_train: 1.9348 acc_train: 0.6590 loss_val: 1.9926 acc_val: 0.5358 time: 0.0040s
Epoch: 0012 loss_train: 1.9113 acc_train: 0.6663 loss_val: 1.9762 acc_val: 0.5431 time: 0.0040s
Epoch: 0013 loss_train: 1.8883 acc_train: 0.6650 loss_val: 1.9600 acc_val: 0.5437 time: 0.0040s
Epoch: 0014 loss_train: 1.8656 acc_train: 0.6699 loss_val: 1.9441 acc_val: 0.5437 time: 0.0040s
Epoch: 0015 loss_train: 1.8433 acc_train: 0.6748 loss_val: 1.9285 acc_val: 0.5455 time: 0.0040s
Epoch: 0016 loss_train: 1.8213 acc_train: 0.6820 loss_val: 1.9131 acc_val: 0.5498 time: 0.0030s
Epoch: 0017 loss_train: 1.7998 acc_train: 0.6917 loss_val: 1.8981 acc_val: 0.5504 time: 0.0040s
Epoch: 0018 loss_train: 1.7786 acc_train: 0.6978 loss_val: 1.8833 acc_val: 0.5552 time: 0.0040s
Epoch: 0019 loss_train: 1.7578 acc_train: 0.7002 loss_val: 1.8688 acc_val: 0.5558 time: 0.0040s
Epoch: 0020 loss_train: 1.7373 acc_train: 0.7015 loss_val: 1.8545 acc_val: 0.5589 time: 0.0040s
Epoch: 0021 loss_train: 1.7173 acc_train: 0.7027 loss_val: 1.8405 acc_val: 0.5643 time: 0.0040s
Epoch: 0022 loss_train: 1.6976 acc_train: 0.7015 loss_val: 1.8268 acc_val: 0.5661 time: 0.0040s
Epoch: 0023 loss_train: 1.6782 acc_train: 0.7015 loss_val: 1.8133 acc_val: 0.5692 time: 0.0050s
Epoch: 0024 loss_train: 1.6592 acc_train: 0.7027 loss_val: 1.8001 acc_val: 0.5704 time: 0.0040s
Epoch: 0025 loss_train: 1.6406 acc_train: 0.7039 loss_val: 1.7871 acc_val: 0.5698 time: 0.0050s
Epoch: 0026 loss_train: 1.6223 acc_train: 0.7039 loss_val: 1.7744 acc_val: 0.5716 time: 0.0040s
Epoch: 0027 loss_train: 1.6044 acc_train: 0.7112 loss_val: 1.7619 acc_val: 0.5746 time: 0.0040s
Epoch: 0028 loss_train: 1.5868 acc_train: 0.7112 loss_val: 1.7497 acc_val: 0.5771 time: 0.0070s
Epoch: 0029 loss_train: 1.5695 acc_train: 0.7124 loss_val: 1.7377 acc_val: 0.5771 time: 0.0070s
Epoch: 0030 loss_train: 1.5526 acc_train: 0.7184 loss_val: 1.7259 acc_val: 0.5777 time: 0.0080s
Epoch: 0031 loss_train: 1.5360 acc_train: 0.7197 loss_val: 1.7144 acc_val: 0.5777 time: 0.0050s
Epoch: 0032 loss_train: 1.5197 acc_train: 0.7209 loss_val: 1.7030 acc_val: 0.5813 time: 0.0050s
Epoch: 0033 loss_train: 1.5038 acc_train: 0.7221 loss_val: 1.6919 acc_val: 0.5813 time: 0.0040s
Epoch: 0034 loss_train: 1.4881 acc_train: 0.7233 loss_val: 1.6811 acc_val: 0.5837 time: 0.0040s
Epoch: 0035 loss_train: 1.4728 acc_train: 0.7245 loss_val: 1.6704 acc_val: 0.5856 time: 0.0040s
Epoch: 0036 loss_train: 1.4577 acc_train: 0.7294 loss_val: 1.6599 acc_val: 0.5862 time: 0.0040s
Epoch: 0037 loss_train: 1.4430 acc_train: 0.7306 loss_val: 1.6497 acc_val: 0.5868 time: 0.0050s
Epoch: 0038 loss_train: 1.4285 acc_train: 0.7354 loss_val: 1.6397 acc_val: 0.5886 time: 0.0060s
Epoch: 0039 loss_train: 1.4143 acc_train: 0.7379 loss_val: 1.6298 acc_val: 0.5892 time: 0.0060s
Epoch: 0040 loss_train: 1.4004 acc_train: 0.7415 loss_val: 1.6202 acc_val: 0.5904 time: 0.0060s
Epoch: 0041 loss_train: 1.3868 acc_train: 0.7451 loss_val: 1.6108 acc_val: 0.5916 time: 0.0060s
Epoch: 0042 loss_train: 1.3735 acc_train: 0.7464 loss_val: 1.6015 acc_val: 0.5904 time: 0.0050s
Epoch: 0043 loss_train: 1.3604 acc_train: 0.7476 loss_val: 1.5925 acc_val: 0.5922 time: 0.0050s
Epoch: 0044 loss_train: 1.3476 acc_train: 0.7488 loss_val: 1.5836 acc_val: 0.5916 time: 0.0040s
Epoch: 0045 loss_train: 1.3350 acc_train: 0.7524 loss_val: 1.5749 acc_val: 0.5922 time: 0.0040s
Epoch: 0046 loss_train: 1.3226 acc_train: 0.7512 loss_val: 1.5664 acc_val: 0.5959 time: 0.0060s
Epoch: 0047 loss_train: 1.3105 acc_train: 0.7524 loss_val: 1.5580 acc_val: 0.5965 time: 0.0050s
Epoch: 0048 loss_train: 1.2987 acc_train: 0.7536 loss_val: 1.5498 acc_val: 0.5965 time: 0.0040s
Epoch: 0049 loss_train: 1.2871 acc_train: 0.7549 loss_val: 1.5418 acc_val: 0.5977 time: 0.0040s
Epoch: 0050 loss_train: 1.2757 acc_train: 0.7561 loss_val: 1.5340 acc_val: 0.5989 time: 0.0050s
Epoch: 0051 loss_train: 1.2645 acc_train: 0.7585 loss_val: 1.5263 acc_val: 0.6007 time: 0.0040s
Epoch: 0052 loss_train: 1.2535 acc_train: 0.7597 loss_val: 1.5187 acc_val: 0.6013 time: 0.0040s
Epoch: 0053 loss_train: 1.2427 acc_train: 0.7621 loss_val: 1.5113 acc_val: 0.6013 time: 0.0040s
Epoch: 0054 loss_train: 1.2322 acc_train: 0.7609 loss_val: 1.5040 acc_val: 0.6025 time: 0.0040s
Epoch: 0055 loss_train: 1.2218 acc_train: 0.7621 loss_val: 1.4969 acc_val: 0.6032 time: 0.0030s
Epoch: 0056 loss_train: 1.2116 acc_train: 0.7633 loss_val: 1.4899 acc_val: 0.6032 time: 0.0040s
Epoch: 0057 loss_train: 1.2016 acc_train: 0.7646 loss_val: 1.4831 acc_val: 0.6025 time: 0.0040s
Epoch: 0058 loss_train: 1.1918 acc_train: 0.7706 loss_val: 1.4764 acc_val: 0.6032 time: 0.0040s
Epoch: 0059 loss_train: 1.1822 acc_train: 0.7718 loss_val: 1.4698 acc_val: 0.6044 time: 0.0040s
Epoch: 0060 loss_train: 1.1727 acc_train: 0.7743 loss_val: 1.4633 acc_val: 0.6056 time: 0.0040s
Epoch: 0061 loss_train: 1.1635 acc_train: 0.7743 loss_val: 1.4569 acc_val: 0.6080 time: 0.0030s
Epoch: 0062 loss_train: 1.1543 acc_train: 0.7767 loss_val: 1.4507 acc_val: 0.6086 time: 0.0030s
Epoch: 0063 loss_train: 1.1454 acc_train: 0.7779 loss_val: 1.4446 acc_val: 0.6098 time: 0.0040s
Epoch: 0064 loss_train: 1.1366 acc_train: 0.7816 loss_val: 1.4386 acc_val: 0.6110 time: 0.0040s
Epoch: 0065 loss_train: 1.1279 acc_train: 0.7828 loss_val: 1.4327 acc_val: 0.6110 time: 0.0040s
Epoch: 0066 loss_train: 1.1194 acc_train: 0.7840 loss_val: 1.4269 acc_val: 0.6135 time: 0.0040s
Epoch: 0067 loss_train: 1.1110 acc_train: 0.7852 loss_val: 1.4212 acc_val: 0.6141 time: 0.0040s
Epoch: 0068 loss_train: 1.1028 acc_train: 0.7852 loss_val: 1.4156 acc_val: 0.6159 time: 0.0040s
Epoch: 0069 loss_train: 1.0948 acc_train: 0.7876 loss_val: 1.4101 acc_val: 0.6165 time: 0.0040s
Epoch: 0070 loss_train: 1.0868 acc_train: 0.7864 loss_val: 1.4047 acc_val: 0.6183 time: 0.0040s
Epoch: 0071 loss_train: 1.0790 acc_train: 0.7864 loss_val: 1.3994 acc_val: 0.6195 time: 0.0040s
Epoch: 0072 loss_train: 1.0713 acc_train: 0.7876 loss_val: 1.3942 acc_val: 0.6220 time: 0.0040s
Epoch: 0073 loss_train: 1.0637 acc_train: 0.7925 loss_val: 1.3891 acc_val: 0.6214 time: 0.0040s
Epoch: 0074 loss_train: 1.0563 acc_train: 0.7937 loss_val: 1.3841 acc_val: 0.6232 time: 0.0040s
Epoch: 0075 loss_train: 1.0490 acc_train: 0.7973 loss_val: 1.3791 acc_val: 0.6232 time: 0.0030s
Epoch: 0076 loss_train: 1.0418 acc_train: 0.7973 loss_val: 1.3743 acc_val: 0.6226 time: 0.0040s
Epoch: 0077 loss_train: 1.0347 acc_train: 0.7985 loss_val: 1.3695 acc_val: 0.6220 time: 0.0040s
Epoch: 0078 loss_train: 1.0277 acc_train: 0.7973 loss_val: 1.3648 acc_val: 0.6226 time: 0.0040s
Epoch: 0079 loss_train: 1.0208 acc_train: 0.7985 loss_val: 1.3602 acc_val: 0.6244 time: 0.0060s
Epoch: 0080 loss_train: 1.0141 acc_train: 0.7998 loss_val: 1.3556 acc_val: 0.6256 time: 0.0080s
Epoch: 0081 loss_train: 1.0074 acc_train: 0.8010 loss_val: 1.3512 acc_val: 0.6244 time: 0.0060s
Epoch: 0082 loss_train: 1.0009 acc_train: 0.8034 loss_val: 1.3468 acc_val: 0.6262 time: 0.0050s
Epoch: 0083 loss_train: 0.9944 acc_train: 0.8046 loss_val: 1.3425 acc_val: 0.6280 time: 0.0040s
Epoch: 0084 loss_train: 0.9880 acc_train: 0.8046 loss_val: 1.3382 acc_val: 0.6280 time: 0.0040s
Epoch: 0085 loss_train: 0.9818 acc_train: 0.8046 loss_val: 1.3341 acc_val: 0.6292 time: 0.0040s
Epoch: 0086 loss_train: 0.9756 acc_train: 0.8070 loss_val: 1.3300 acc_val: 0.6323 time: 0.0040s
Epoch: 0087 loss_train: 0.9695 acc_train: 0.8083 loss_val: 1.3259 acc_val: 0.6329 time: 0.0030s
Epoch: 0088 loss_train: 0.9635 acc_train: 0.8083 loss_val: 1.3220 acc_val: 0.6341 time: 0.0040s
Epoch: 0089 loss_train: 0.9576 acc_train: 0.8070 loss_val: 1.3181 acc_val: 0.6347 time: 0.0040s
Epoch: 0090 loss_train: 0.9517 acc_train: 0.8083 loss_val: 1.3142 acc_val: 0.6353 time: 0.0040s
Epoch: 0091 loss_train: 0.9460 acc_train: 0.8083 loss_val: 1.3104 acc_val: 0.6371 time: 0.0030s
Epoch: 0092 loss_train: 0.9403 acc_train: 0.8083 loss_val: 1.3067 acc_val: 0.6377 time: 0.0030s
Epoch: 0093 loss_train: 0.9347 acc_train: 0.8083 loss_val: 1.3031 acc_val: 0.6383 time: 0.0040s
Epoch: 0094 loss_train: 0.9292 acc_train: 0.8070 loss_val: 1.2995 acc_val: 0.6390 time: 0.0040s
Epoch: 0095 loss_train: 0.9238 acc_train: 0.8070 loss_val: 1.2959 acc_val: 0.6396 time: 0.0040s
Epoch: 0096 loss_train: 0.9184 acc_train: 0.8070 loss_val: 1.2925 acc_val: 0.6396 time: 0.0030s
Epoch: 0097 loss_train: 0.9132 acc_train: 0.8083 loss_val: 1.2890 acc_val: 0.6414 time: 0.0040s
Epoch: 0098 loss_train: 0.9079 acc_train: 0.8095 loss_val: 1.2857 acc_val: 0.6426 time: 0.0040s
Epoch: 0099 loss_train: 0.9028 acc_train: 0.8119 loss_val: 1.2824 acc_val: 0.6426 time: 0.0040s
Epoch: 0100 loss_train: 0.8977 acc_train: 0.8119 loss_val: 1.2791 acc_val: 0.6432 time: 0.0040s
Epoch: 0101 loss_train: 0.8927 acc_train: 0.8119 loss_val: 1.2759 acc_val: 0.6438 time: 0.0030s
Epoch: 0102 loss_train: 0.8878 acc_train: 0.8143 loss_val: 1.2727 acc_val: 0.6450 time: 0.0040s
Epoch: 0103 loss_train: 0.8829 acc_train: 0.8155 loss_val: 1.2696 acc_val: 0.6456 time: 0.0040s
Epoch: 0104 loss_train: 0.8781 acc_train: 0.8155 loss_val: 1.2666 acc_val: 0.6456 time: 0.0040s
Epoch: 0105 loss_train: 0.8733 acc_train: 0.8167 loss_val: 1.2636 acc_val: 0.6456 time: 0.0030s
Epoch: 0106 loss_train: 0.8686 acc_train: 0.8167 loss_val: 1.2606 acc_val: 0.6462 time: 0.0030s
Epoch: 0107 loss_train: 0.8640 acc_train: 0.8192 loss_val: 1.2577 acc_val: 0.6450 time: 0.0070s
Epoch: 0108 loss_train: 0.8594 acc_train: 0.8216 loss_val: 1.2548 acc_val: 0.6456 time: 0.0070s
Epoch: 0109 loss_train: 0.8549 acc_train: 0.8216 loss_val: 1.2520 acc_val: 0.6456 time: 0.0070s
Epoch: 0110 loss_train: 0.8504 acc_train: 0.8216 loss_val: 1.2492 acc_val: 0.6450 time: 0.0060s
Epoch: 0111 loss_train: 0.8460 acc_train: 0.8216 loss_val: 1.2465 acc_val: 0.6450 time: 0.0050s
Epoch: 0112 loss_train: 0.8416 acc_train: 0.8228 loss_val: 1.2438 acc_val: 0.6450 time: 0.0040s
Epoch: 0113 loss_train: 0.8373 acc_train: 0.8228 loss_val: 1.2411 acc_val: 0.6450 time: 0.0030s
Epoch: 0114 loss_train: 0.8331 acc_train: 0.8228 loss_val: 1.2385 acc_val: 0.6450 time: 0.0050s
Epoch: 0115 loss_train: 0.8289 acc_train: 0.8228 loss_val: 1.2360 acc_val: 0.6450 time: 0.0030s
Epoch: 0116 loss_train: 0.8247 acc_train: 0.8240 loss_val: 1.2334 acc_val: 0.6444 time: 0.0040s
Epoch: 0117 loss_train: 0.8206 acc_train: 0.8252 loss_val: 1.2310 acc_val: 0.6444 time: 0.0040s
Epoch: 0118 loss_train: 0.8166 acc_train: 0.8252 loss_val: 1.2285 acc_val: 0.6438 time: 0.0040s
Epoch: 0119 loss_train: 0.8126 acc_train: 0.8252 loss_val: 1.2261 acc_val: 0.6450 time: 0.0040s
Epoch: 0120 loss_train: 0.8086 acc_train: 0.8252 loss_val: 1.2237 acc_val: 0.6444 time: 0.0030s
Epoch: 0121 loss_train: 0.8047 acc_train: 0.8252 loss_val: 1.2214 acc_val: 0.6462 time: 0.0030s
Epoch: 0122 loss_train: 0.8008 acc_train: 0.8265 loss_val: 1.2191 acc_val: 0.6462 time: 0.0030s
Epoch: 0123 loss_train: 0.7970 acc_train: 0.8265 loss_val: 1.2169 acc_val: 0.6456 time: 0.0030s
Epoch: 0124 loss_train: 0.7932 acc_train: 0.8265 loss_val: 1.2146 acc_val: 0.6456 time: 0.0030s
Epoch: 0125 loss_train: 0.7895 acc_train: 0.8265 loss_val: 1.2125 acc_val: 0.6468 time: 0.0030s
Epoch: 0126 loss_train: 0.7858 acc_train: 0.8277 loss_val: 1.2103 acc_val: 0.6475 time: 0.0040s
Epoch: 0127 loss_train: 0.7821 acc_train: 0.8301 loss_val: 1.2082 acc_val: 0.6475 time: 0.0030s
Epoch: 0128 loss_train: 0.7785 acc_train: 0.8301 loss_val: 1.2061 acc_val: 0.6468 time: 0.0030s
Epoch: 0129 loss_train: 0.7749 acc_train: 0.8301 loss_val: 1.2040 acc_val: 0.6468 time: 0.0030s
Epoch: 0130 loss_train: 0.7714 acc_train: 0.8301 loss_val: 1.2020 acc_val: 0.6475 time: 0.0040s
Epoch: 0131 loss_train: 0.7679 acc_train: 0.8301 loss_val: 1.2000 acc_val: 0.6487 time: 0.0030s
Epoch: 0132 loss_train: 0.7644 acc_train: 0.8301 loss_val: 1.1981 acc_val: 0.6505 time: 0.0030s
Epoch: 0133 loss_train: 0.7610 acc_train: 0.8301 loss_val: 1.1961 acc_val: 0.6499 time: 0.0040s
Epoch: 0134 loss_train: 0.7576 acc_train: 0.8313 loss_val: 1.1942 acc_val: 0.6505 time: 0.0030s
Epoch: 0135 loss_train: 0.7542 acc_train: 0.8337 loss_val: 1.1924 acc_val: 0.6517 time: 0.0030s
Epoch: 0136 loss_train: 0.7509 acc_train: 0.8337 loss_val: 1.1905 acc_val: 0.6517 time: 0.0040s
Epoch: 0137 loss_train: 0.7476 acc_train: 0.8337 loss_val: 1.1887 acc_val: 0.6529 time: 0.0040s
Epoch: 0138 loss_train: 0.7444 acc_train: 0.8337 loss_val: 1.1869 acc_val: 0.6541 time: 0.0030s
Epoch: 0139 loss_train: 0.7412 acc_train: 0.8350 loss_val: 1.1852 acc_val: 0.6547 time: 0.0040s
Epoch: 0140 loss_train: 0.7380 acc_train: 0.8362 loss_val: 1.1835 acc_val: 0.6547 time: 0.0040s
Epoch: 0141 loss_train: 0.7348 acc_train: 0.8362 loss_val: 1.1818 acc_val: 0.6547 time: 0.0040s
Epoch: 0142 loss_train: 0.7317 acc_train: 0.8362 loss_val: 1.1801 acc_val: 0.6547 time: 0.0090s
Epoch: 0143 loss_train: 0.7286 acc_train: 0.8362 loss_val: 1.1785 acc_val: 0.6547 time: 0.0050s
Epoch: 0144 loss_train: 0.7256 acc_train: 0.8398 loss_val: 1.1768 acc_val: 0.6553 time: 0.0030s
Epoch: 0145 loss_train: 0.7226 acc_train: 0.8410 loss_val: 1.1753 acc_val: 0.6559 time: 0.0040s
Epoch: 0146 loss_train: 0.7196 acc_train: 0.8410 loss_val: 1.1737 acc_val: 0.6566 time: 0.0040s
Epoch: 0147 loss_train: 0.7166 acc_train: 0.8410 loss_val: 1.1721 acc_val: 0.6572 time: 0.0040s
Epoch: 0148 loss_train: 0.7136 acc_train: 0.8422 loss_val: 1.1706 acc_val: 0.6572 time: 0.0030s
Epoch: 0149 loss_train: 0.7107 acc_train: 0.8434 loss_val: 1.1691 acc_val: 0.6572 time: 0.0030s
Epoch: 0150 loss_train: 0.7079 acc_train: 0.8459 loss_val: 1.1677 acc_val: 0.6566 time: 0.0040s
Epoch: 0151 loss_train: 0.7050 acc_train: 0.8459 loss_val: 1.1662 acc_val: 0.6566 time: 0.0030s
Epoch: 0152 loss_train: 0.7022 acc_train: 0.8459 loss_val: 1.1648 acc_val: 0.6559 time: 0.0030s
Epoch: 0153 loss_train: 0.6994 acc_train: 0.8471 loss_val: 1.1634 acc_val: 0.6572 time: 0.0040s
Epoch: 0154 loss_train: 0.6966 acc_train: 0.8471 loss_val: 1.1620 acc_val: 0.6578 time: 0.0040s
Epoch: 0155 loss_train: 0.6938 acc_train: 0.8471 loss_val: 1.1607 acc_val: 0.6578 time: 0.0040s
Epoch: 0156 loss_train: 0.6911 acc_train: 0.8471 loss_val: 1.1593 acc_val: 0.6578 time: 0.0030s
Epoch: 0157 loss_train: 0.6884 acc_train: 0.8471 loss_val: 1.1580 acc_val: 0.6578 time: 0.0040s
Epoch: 0158 loss_train: 0.6858 acc_train: 0.8471 loss_val: 1.1567 acc_val: 0.6584 time: 0.0040s
Epoch: 0159 loss_train: 0.6831 acc_train: 0.8471 loss_val: 1.1555 acc_val: 0.6584 time: 0.0040s
Epoch: 0160 loss_train: 0.6805 acc_train: 0.8471 loss_val: 1.1542 acc_val: 0.6584 time: 0.0030s
Epoch: 0161 loss_train: 0.6779 acc_train: 0.8483 loss_val: 1.1530 acc_val: 0.6578 time: 0.0040s
Epoch: 0162 loss_train: 0.6753 acc_train: 0.8483 loss_val: 1.1518 acc_val: 0.6584 time: 0.0040s
Epoch: 0163 loss_train: 0.6728 acc_train: 0.8495 loss_val: 1.1506 acc_val: 0.6584 time: 0.0040s
Epoch: 0164 loss_train: 0.6702 acc_train: 0.8495 loss_val: 1.1494 acc_val: 0.6578 time: 0.0030s
Epoch: 0165 loss_train: 0.6677 acc_train: 0.8495 loss_val: 1.1483 acc_val: 0.6584 time: 0.0040s
Epoch: 0166 loss_train: 0.6652 acc_train: 0.8507 loss_val: 1.1472 acc_val: 0.6584 time: 0.0040s
Epoch: 0167 loss_train: 0.6628 acc_train: 0.8519 loss_val: 1.1461 acc_val: 0.6584 time: 0.0030s
Epoch: 0168 loss_train: 0.6603 acc_train: 0.8519 loss_val: 1.1450 acc_val: 0.6584 time: 0.0040s
Epoch: 0169 loss_train: 0.6579 acc_train: 0.8519 loss_val: 1.1439 acc_val: 0.6584 time: 0.0030s
Epoch: 0170 loss_train: 0.6555 acc_train: 0.8519 loss_val: 1.1428 acc_val: 0.6584 time: 0.0030s
Epoch: 0171 loss_train: 0.6531 acc_train: 0.8519 loss_val: 1.1418 acc_val: 0.6584 time: 0.0030s
Epoch: 0172 loss_train: 0.6508 acc_train: 0.8532 loss_val: 1.1408 acc_val: 0.6584 time: 0.0030s
Epoch: 0173 loss_train: 0.6484 acc_train: 0.8532 loss_val: 1.1398 acc_val: 0.6584 time: 0.0044s
Epoch: 0174 loss_train: 0.6461 acc_train: 0.8544 loss_val: 1.1388 acc_val: 0.6572 time: 0.0025s
Epoch: 0175 loss_train: 0.6438 acc_train: 0.8544 loss_val: 1.1378 acc_val: 0.6572 time: 0.0040s
Epoch: 0176 loss_train: 0.6415 acc_train: 0.8544 loss_val: 1.1369 acc_val: 0.6566 time: 0.0030s
Epoch: 0177 loss_train: 0.6393 acc_train: 0.8532 loss_val: 1.1359 acc_val: 0.6566 time: 0.0040s
Epoch: 0178 loss_train: 0.6370 acc_train: 0.8532 loss_val: 1.1350 acc_val: 0.6566 time: 0.0040s
Epoch: 0179 loss_train: 0.6348 acc_train: 0.8532 loss_val: 1.1341 acc_val: 0.6566 time: 0.0030s
Epoch: 0180 loss_train: 0.6326 acc_train: 0.8532 loss_val: 1.1332 acc_val: 0.6572 time: 0.0040s
Epoch: 0181 loss_train: 0.6304 acc_train: 0.8532 loss_val: 1.1324 acc_val: 0.6572 time: 0.0030s
Epoch: 0182 loss_train: 0.6282 acc_train: 0.8532 loss_val: 1.1315 acc_val: 0.6566 time: 0.0030s
Epoch: 0183 loss_train: 0.6261 acc_train: 0.8532 loss_val: 1.1307 acc_val: 0.6572 time: 0.0060s
Epoch: 0184 loss_train: 0.6239 acc_train: 0.8544 loss_val: 1.1298 acc_val: 0.6584 time: 0.0070s
Epoch: 0185 loss_train: 0.6218 acc_train: 0.8568 loss_val: 1.1290 acc_val: 0.6584 time: 0.0050s
Epoch: 0186 loss_train: 0.6197 acc_train: 0.8568 loss_val: 1.1282 acc_val: 0.6584 time: 0.0040s
Epoch: 0187 loss_train: 0.6176 acc_train: 0.8568 loss_val: 1.1274 acc_val: 0.6584 time: 0.0040s
Epoch: 0188 loss_train: 0.6156 acc_train: 0.8580 loss_val: 1.1267 acc_val: 0.6584 time: 0.0040s
Epoch: 0189 loss_train: 0.6135 acc_train: 0.8604 loss_val: 1.1259 acc_val: 0.6584 time: 0.0030s
Epoch: 0190 loss_train: 0.6115 acc_train: 0.8617 loss_val: 1.1252 acc_val: 0.6584 time: 0.0040s
Epoch: 0191 loss_train: 0.6094 acc_train: 0.8641 loss_val: 1.1244 acc_val: 0.6590 time: 0.0040s
Epoch: 0192 loss_train: 0.6074 acc_train: 0.8641 loss_val: 1.1237 acc_val: 0.6596 time: 0.0040s
Epoch: 0193 loss_train: 0.6054 acc_train: 0.8641 loss_val: 1.1230 acc_val: 0.6596 time: 0.0030s
Epoch: 0194 loss_train: 0.6035 acc_train: 0.8641 loss_val: 1.1223 acc_val: 0.6596 time: 0.0040s
Epoch: 0195 loss_train: 0.6015 acc_train: 0.8641 loss_val: 1.1217 acc_val: 0.6590 time: 0.0040s
Epoch: 0196 loss_train: 0.5996 acc_train: 0.8641 loss_val: 1.1210 acc_val: 0.6602 time: 0.0040s
Epoch: 0197 loss_train: 0.5976 acc_train: 0.8641 loss_val: 1.1203 acc_val: 0.6608 time: 0.0040s
Epoch: 0198 loss_train: 0.5957 acc_train: 0.8653 loss_val: 1.1197 acc_val: 0.6608 time: 0.0040s
Epoch: 0199 loss_train: 0.5938 acc_train: 0.8665 loss_val: 1.1191 acc_val: 0.6608 time: 0.0030s
Epoch: 0200 loss_train: 0.5919 acc_train: 0.8665 loss_val: 1.1184 acc_val: 0.6602 time: 0.0040s
Epoch: 0201 loss_train: 0.5901 acc_train: 0.8665 loss_val: 1.1178 acc_val: 0.6602 time: 0.0040s
Epoch: 0202 loss_train: 0.5882 acc_train: 0.8677 loss_val: 1.1173 acc_val: 0.6602 time: 0.0040s
Epoch: 0203 loss_train: 0.5864 acc_train: 0.8677 loss_val: 1.1167 acc_val: 0.6608 time: 0.0050s
Epoch: 0204 loss_train: 0.5845 acc_train: 0.8689 loss_val: 1.1161 acc_val: 0.6602 time: 0.0040s
Epoch: 0205 loss_train: 0.5827 acc_train: 0.8701 loss_val: 1.1155 acc_val: 0.6596 time: 0.0030s
Epoch: 0206 loss_train: 0.5809 acc_train: 0.8714 loss_val: 1.1150 acc_val: 0.6590 time: 0.0040s
Epoch: 0207 loss_train: 0.5791 acc_train: 0.8714 loss_val: 1.1145 acc_val: 0.6590 time: 0.0040s
Epoch: 0208 loss_train: 0.5773 acc_train: 0.8726 loss_val: 1.1139 acc_val: 0.6590 time: 0.0040s
Epoch: 0209 loss_train: 0.5756 acc_train: 0.8738 loss_val: 1.1134 acc_val: 0.6602 time: 0.0060s
Epoch: 0210 loss_train: 0.5738 acc_train: 0.8750 loss_val: 1.1129 acc_val: 0.6602 time: 0.0070s
Epoch: 0211 loss_train: 0.5721 acc_train: 0.8750 loss_val: 1.1124 acc_val: 0.6602 time: 0.0070s
Epoch: 0212 loss_train: 0.5703 acc_train: 0.8750 loss_val: 1.1119 acc_val: 0.6602 time: 0.0070s
Epoch: 0213 loss_train: 0.5686 acc_train: 0.8762 loss_val: 1.1115 acc_val: 0.6596 time: 0.0050s
Epoch: 0214 loss_train: 0.5669 acc_train: 0.8774 loss_val: 1.1110 acc_val: 0.6596 time: 0.0050s
Epoch: 0215 loss_train: 0.5652 acc_train: 0.8786 loss_val: 1.1105 acc_val: 0.6596 time: 0.0040s
Epoch: 0216 loss_train: 0.5635 acc_train: 0.8799 loss_val: 1.1101 acc_val: 0.6596 time: 0.0030s
Epoch: 0217 loss_train: 0.5619 acc_train: 0.8799 loss_val: 1.1097 acc_val: 0.6602 time: 0.0040s
Epoch: 0218 loss_train: 0.5602 acc_train: 0.8799 loss_val: 1.1092 acc_val: 0.6608 time: 0.0040s
Epoch: 0219 loss_train: 0.5586 acc_train: 0.8811 loss_val: 1.1088 acc_val: 0.6608 time: 0.0040s
Epoch: 0220 loss_train: 0.5569 acc_train: 0.8811 loss_val: 1.1084 acc_val: 0.6614 time: 0.0050s
Epoch: 0221 loss_train: 0.5553 acc_train: 0.8811 loss_val: 1.1080 acc_val: 0.6614 time: 0.0070s
Epoch: 0222 loss_train: 0.5537 acc_train: 0.8811 loss_val: 1.1076 acc_val: 0.6620 time: 0.0080s
Epoch: 0223 loss_train: 0.5521 acc_train: 0.8823 loss_val: 1.1072 acc_val: 0.6614 time: 0.0070s
Epoch: 0224 loss_train: 0.5505 acc_train: 0.8823 loss_val: 1.1069 acc_val: 0.6614 time: 0.0060s
Epoch: 0225 loss_train: 0.5489 acc_train: 0.8823 loss_val: 1.1065 acc_val: 0.6608 time: 0.0040s
Epoch: 0226 loss_train: 0.5473 acc_train: 0.8823 loss_val: 1.1061 acc_val: 0.6608 time: 0.0050s
Epoch: 0227 loss_train: 0.5458 acc_train: 0.8823 loss_val: 1.1058 acc_val: 0.6608 time: 0.0040s
Epoch: 0228 loss_train: 0.5442 acc_train: 0.8835 loss_val: 1.1055 acc_val: 0.6608 time: 0.0040s
Epoch: 0229 loss_train: 0.5427 acc_train: 0.8835 loss_val: 1.1051 acc_val: 0.6608 time: 0.0040s
Epoch: 0230 loss_train: 0.5411 acc_train: 0.8847 loss_val: 1.1048 acc_val: 0.6608 time: 0.0040s
Epoch: 0231 loss_train: 0.5396 acc_train: 0.8847 loss_val: 1.1045 acc_val: 0.6602 time: 0.0040s
Epoch: 0232 loss_train: 0.5381 acc_train: 0.8847 loss_val: 1.1042 acc_val: 0.6608 time: 0.0030s
Epoch: 0233 loss_train: 0.5366 acc_train: 0.8847 loss_val: 1.1039 acc_val: 0.6614 time: 0.0040s
Epoch: 0234 loss_train: 0.5351 acc_train: 0.8847 loss_val: 1.1036 acc_val: 0.6614 time: 0.0040s
Epoch: 0235 loss_train: 0.5336 acc_train: 0.8859 loss_val: 1.1033 acc_val: 0.6614 time: 0.0040s
Epoch: 0236 loss_train: 0.5322 acc_train: 0.8871 loss_val: 1.1030 acc_val: 0.6614 time: 0.0030s
Epoch: 0237 loss_train: 0.5307 acc_train: 0.8871 loss_val: 1.1028 acc_val: 0.6620 time: 0.0040s
Epoch: 0238 loss_train: 0.5292 acc_train: 0.8871 loss_val: 1.1025 acc_val: 0.6620 time: 0.0040s
Epoch: 0239 loss_train: 0.5278 acc_train: 0.8871 loss_val: 1.1023 acc_val: 0.6620 time: 0.0040s
Epoch: 0240 loss_train: 0.5264 acc_train: 0.8871 loss_val: 1.1020 acc_val: 0.6620 time: 0.0030s
Epoch: 0241 loss_train: 0.5249 acc_train: 0.8871 loss_val: 1.1018 acc_val: 0.6626 time: 0.0040s
Epoch: 0242 loss_train: 0.5235 acc_train: 0.8871 loss_val: 1.1016 acc_val: 0.6626 time: 0.0040s
Epoch: 0243 loss_train: 0.5221 acc_train: 0.8871 loss_val: 1.1013 acc_val: 0.6626 time: 0.0040s
Epoch: 0244 loss_train: 0.5207 acc_train: 0.8871 loss_val: 1.1011 acc_val: 0.6626 time: 0.0030s
Epoch: 0245 loss_train: 0.5193 acc_train: 0.8871 loss_val: 1.1009 acc_val: 0.6626 time: 0.0040s
Epoch: 0246 loss_train: 0.5179 acc_train: 0.8883 loss_val: 1.1007 acc_val: 0.6626 time: 0.0040s
Epoch: 0247 loss_train: 0.5165 acc_train: 0.8883 loss_val: 1.1005 acc_val: 0.6626 time: 0.0040s
Epoch: 0248 loss_train: 0.5152 acc_train: 0.8871 loss_val: 1.1003 acc_val: 0.6626 time: 0.0040s
Epoch: 0249 loss_train: 0.5138 acc_train: 0.8871 loss_val: 1.1001 acc_val: 0.6620 time: 0.0040s
Epoch: 0250 loss_train: 0.5125 acc_train: 0.8871 loss_val: 1.0999 acc_val: 0.6614 time: 0.0030s
Epoch: 0251 loss_train: 0.5111 acc_train: 0.8883 loss_val: 1.0998 acc_val: 0.6614 time: 0.0040s
Epoch: 0252 loss_train: 0.5098 acc_train: 0.8883 loss_val: 1.0996 acc_val: 0.6614 time: 0.0040s
Epoch: 0253 loss_train: 0.5085 acc_train: 0.8883 loss_val: 1.0995 acc_val: 0.6620 time: 0.0040s
Epoch: 0254 loss_train: 0.5071 acc_train: 0.8883 loss_val: 1.0993 acc_val: 0.6620 time: 0.0040s
Epoch: 0255 loss_train: 0.5058 acc_train: 0.8883 loss_val: 1.0992 acc_val: 0.6626 time: 0.0040s
Epoch: 0256 loss_train: 0.5045 acc_train: 0.8883 loss_val: 1.0990 acc_val: 0.6632 time: 0.0030s
Epoch: 0257 loss_train: 0.5032 acc_train: 0.8883 loss_val: 1.0989 acc_val: 0.6632 time: 0.0040s
Epoch: 0258 loss_train: 0.5020 acc_train: 0.8883 loss_val: 1.0988 acc_val: 0.6632 time: 0.0040s
Epoch: 0259 loss_train: 0.5007 acc_train: 0.8883 loss_val: 1.0986 acc_val: 0.6632 time: 0.0050s
Epoch: 0260 loss_train: 0.4994 acc_train: 0.8883 loss_val: 1.0985 acc_val: 0.6632 time: 0.0040s
Epoch: 0261 loss_train: 0.4981 acc_train: 0.8883 loss_val: 1.0984 acc_val: 0.6632 time: 0.0040s
Epoch: 0262 loss_train: 0.4969 acc_train: 0.8883 loss_val: 1.0983 acc_val: 0.6632 time: 0.0040s
Epoch: 0263 loss_train: 0.4956 acc_train: 0.8883 loss_val: 1.0982 acc_val: 0.6632 time: 0.0040s
Epoch: 0264 loss_train: 0.4944 acc_train: 0.8883 loss_val: 1.0981 acc_val: 0.6632 time: 0.0050s
Epoch: 0265 loss_train: 0.4931 acc_train: 0.8908 loss_val: 1.0980 acc_val: 0.6632 time: 0.0040s
Epoch: 0266 loss_train: 0.4919 acc_train: 0.8908 loss_val: 1.0979 acc_val: 0.6626 time: 0.0040s
Epoch: 0267 loss_train: 0.4907 acc_train: 0.8908 loss_val: 1.0978 acc_val: 0.6626 time: 0.0040s
Epoch: 0268 loss_train: 0.4895 acc_train: 0.8908 loss_val: 1.0978 acc_val: 0.6626 time: 0.0040s
Epoch: 0269 loss_train: 0.4883 acc_train: 0.8920 loss_val: 1.0977 acc_val: 0.6626 time: 0.0040s
Epoch: 0270 loss_train: 0.4871 acc_train: 0.8920 loss_val: 1.0976 acc_val: 0.6626 time: 0.0050s
Epoch: 0271 loss_train: 0.4859 acc_train: 0.8920 loss_val: 1.0976 acc_val: 0.6626 time: 0.0030s
Epoch: 0272 loss_train: 0.4847 acc_train: 0.8920 loss_val: 1.0975 acc_val: 0.6626 time: 0.0030s
Epoch: 0273 loss_train: 0.4835 acc_train: 0.8920 loss_val: 1.0975 acc_val: 0.6626 time: 0.0030s
Epoch: 0274 loss_train: 0.4823 acc_train: 0.8932 loss_val: 1.0974 acc_val: 0.6620 time: 0.0040s
Epoch: 0275 loss_train: 0.4811 acc_train: 0.8932 loss_val: 1.0974 acc_val: 0.6620 time: 0.0030s
Epoch: 0276 loss_train: 0.4800 acc_train: 0.8932 loss_val: 1.0974 acc_val: 0.6620 time: 0.0030s
Epoch: 0277 loss_train: 0.4788 acc_train: 0.8932 loss_val: 1.0973 acc_val: 0.6620 time: 0.0040s
Epoch: 0278 loss_train: 0.4777 acc_train: 0.8932 loss_val: 1.0973 acc_val: 0.6620 time: 0.0030s
Epoch: 0279 loss_train: 0.4765 acc_train: 0.8932 loss_val: 1.0973 acc_val: 0.6626 time: 0.0040s
Epoch: 0280 loss_train: 0.4754 acc_train: 0.8932 loss_val: 1.0973 acc_val: 0.6626 time: 0.0040s
Epoch: 0281 loss_train: 0.4743 acc_train: 0.8932 loss_val: 1.0973 acc_val: 0.6626 time: 0.0040s
Epoch: 0282 loss_train: 0.4731 acc_train: 0.8944 loss_val: 1.0973 acc_val: 0.6626 time: 0.0040s
Epoch: 0283 loss_train: 0.4720 acc_train: 0.8944 loss_val: 1.0973 acc_val: 0.6632 time: 0.0040s
Epoch: 0284 loss_train: 0.4709 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6632 time: 0.0040s
Epoch: 0285 loss_train: 0.4698 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6632 time: 0.0040s
Epoch: 0286 loss_train: 0.4687 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6632 time: 0.0040s
Epoch: 0287 loss_train: 0.4676 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6632 time: 0.0040s
Epoch: 0288 loss_train: 0.4665 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6638 time: 0.0040s
Epoch: 0289 loss_train: 0.4654 acc_train: 0.8956 loss_val: 1.0973 acc_val: 0.6638 time: 0.0040s
Epoch: 0290 loss_train: 0.4643 acc_train: 0.8968 loss_val: 1.0974 acc_val: 0.6626 time: 0.0050s
Epoch: 0291 loss_train: 0.4633 acc_train: 0.8968 loss_val: 1.0974 acc_val: 0.6614 time: 0.0040s
Epoch: 0292 loss_train: 0.4622 acc_train: 0.8968 loss_val: 1.0974 acc_val: 0.6614 time: 0.0030s
Epoch: 0293 loss_train: 0.4611 acc_train: 0.8968 loss_val: 1.0975 acc_val: 0.6614 time: 0.0030s
Epoch: 0294 loss_train: 0.4601 acc_train: 0.8968 loss_val: 1.0975 acc_val: 0.6620 time: 0.0040s
Epoch: 0295 loss_train: 0.4590 acc_train: 0.8968 loss_val: 1.0976 acc_val: 0.6620 time: 0.0040s
Epoch: 0296 loss_train: 0.4580 acc_train: 0.8968 loss_val: 1.0976 acc_val: 0.6620 time: 0.0040s
Epoch: 0297 loss_train: 0.4569 acc_train: 0.8968 loss_val: 1.0977 acc_val: 0.6620 time: 0.0030s
Epoch: 0298 loss_train: 0.4559 acc_train: 0.8968 loss_val: 1.0977 acc_val: 0.6620 time: 0.0040s
Epoch: 0299 loss_train: 0.4549 acc_train: 0.8968 loss_val: 1.0978 acc_val: 0.6620 time: 0.0030s
Epoch: 0300 loss_train: 0.4538 acc_train: 0.8968 loss_val: 1.0979 acc_val: 0.6620 time: 0.0030s
Epoch: 0001 loss_train: 0.4528 acc_train: 0.8968 loss_val: 1.0979 acc_val: 0.6620 time: 0.0030s
Ranking optimizing... 
Now Average NDCG@k =  0.5587780475616455
Epoch: 0002 loss_train: 0.4519 acc_train: 0.8968 loss_val: 1.0980 acc_val: 0.6632 time: 0.0050s
Ranking optimizing... 
Now Average NDCG@k =  0.5595831274986267
Epoch: 0003 loss_train: 0.4510 acc_train: 0.8956 loss_val: 1.0984 acc_val: 0.6638 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5610162615776062
Epoch: 0004 loss_train: 0.4504 acc_train: 0.8956 loss_val: 1.0989 acc_val: 0.6657 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5629385709762573
Epoch: 0005 loss_train: 0.4499 acc_train: 0.8932 loss_val: 1.0998 acc_val: 0.6650 time: 0.0030s
Ranking optimizing... 
Now Average NDCG@k =  0.5651437044143677
Epoch: 0006 loss_train: 0.4497 acc_train: 0.8932 loss_val: 1.1008 acc_val: 0.6632 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5674173831939697
Epoch: 0007 loss_train: 0.4499 acc_train: 0.8932 loss_val: 1.1021 acc_val: 0.6626 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5700745582580566
Epoch: 0008 loss_train: 0.4504 acc_train: 0.8920 loss_val: 1.1037 acc_val: 0.6632 time: 0.0030s
Ranking optimizing... 
Now Average NDCG@k =  0.5726330876350403
Epoch: 0009 loss_train: 0.4512 acc_train: 0.8920 loss_val: 1.1056 acc_val: 0.6626 time: 0.0667s
Ranking optimizing... 
Now Average NDCG@k =  0.5755036473274231
Epoch: 0010 loss_train: 0.4524 acc_train: 0.8896 loss_val: 1.1079 acc_val: 0.6596 time: 0.0050s
Ranking optimizing... 
Now Average NDCG@k =  0.5782231092453003
Epoch: 0011 loss_train: 0.4539 acc_train: 0.8908 loss_val: 1.1106 acc_val: 0.6590 time: 0.0030s
Ranking optimizing... 
Now Average NDCG@k =  0.5812934041023254
Epoch: 0012 loss_train: 0.4558 acc_train: 0.8920 loss_val: 1.1136 acc_val: 0.6602 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5839889049530029
Epoch: 0013 loss_train: 0.4581 acc_train: 0.8908 loss_val: 1.1171 acc_val: 0.6559 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5867074131965637
Epoch: 0014 loss_train: 0.4607 acc_train: 0.8920 loss_val: 1.1208 acc_val: 0.6541 time: 0.0040s
Ranking optimizing... 
Now Average NDCG@k =  0.5890056490898132
Epoch: 0015 loss_train: 0.4637 acc_train: 0.8908 loss_val: 1.1250 acc_val: 0.6505 time: 0.0030s
Ranking optimizing... 
Now Average NDCG@k =  0.5915500521659851
Test set results: loss= 1.0751 accuracy= 0.6646

Process finished with exit code 0
```

## Log example for link prediction
```
python gcn_feature_NDCG.py
```
Log example:
```
BlogCatalog
Using BlogCatalog dataset
started generating 8587 negative test edges...
started generating 4293 negative validation edges...
Epoch: 0001 train_loss= 1.35043 val_ap= 0.70789 time= 1.51533
Test AUC score: 0.7314498658053916
Test AP score: 0.708957710106427
Epoch: 0002 train_loss= 1.32675 val_ap= 0.75184 time= 0.44504
Test AUC score: 0.7614840948186782
Test AP score: 0.7524043732986254
Epoch: 0003 train_loss= 1.31382 val_ap= 0.77562 time= 0.38231
Test AUC score: 0.7809985978599031
Test AP score: 0.7774743182804028
Epoch: 0004 train_loss= 1.30326 val_ap= 0.78934 time= 0.36539
Test AUC score: 0.7954881315402673
Test AP score: 0.7918687015714113
Epoch: 0005 train_loss= 1.28687 val_ap= 0.80163 time= 0.39028
Test AUC score: 0.8059270631374238
Test AP score: 0.803719288735776
Epoch: 0006 train_loss= 1.27477 val_ap= 0.80946 time= 0.39526
Test AUC score: 0.8121580555233048
Test AP score: 0.8106693401972493
Epoch: 0007 train_loss= 1.26619 val_ap= 0.81227 time= 0.40820
Test AUC score: 0.8148095824746062
Test AP score: 0.8125941673699439
Epoch: 0008 train_loss= 1.25778 val_ap= 0.81240 time= 0.38132
Test AUC score: 0.8154717573040319
Test AP score: 0.8126091609083226
Epoch: 0009 train_loss= 1.24978 val_ap= 0.81195 time= 0.42712
Test AUC score: 0.8154907031272367
Test AP score: 0.8128655076978454
Epoch: 0010 train_loss= 1.24401 val_ap= 0.81101 time= 0.44205
Test AUC score: 0.8151635316799185
Test AP score: 0.8132444110486385
Epoch: 0011 train_loss= 1.23968 val_ap= 0.80932 time= 0.40024
Test AUC score: 0.8142504365778126
Test AP score: 0.8131172487673366
Epoch: 0012 train_loss= 1.23549 val_ap= 0.80637 time= 0.46396
Test AUC score: 0.8121455718939133
Test AP score: 0.811870176330493
Epoch: 0013 train_loss= 1.23233 val_ap= 0.80267 time= 0.39327
Test AUC score: 0.8093560130252331
Test AP score: 0.809678958074988
Epoch: 0014 train_loss= 1.23049 val_ap= 0.79927 time= 0.39227
Test AUC score: 0.8066822922015804
Test AP score: 0.8073087563361425
Epoch: 0015 train_loss= 1.22892 val_ap= 0.79770 time= 0.41119
Test AUC score: 0.8050427461576087
Test AP score: 0.8059080911032398
Epoch: 0016 train_loss= 1.22686 val_ap= 0.79805 time= 0.42911
Test AUC score: 0.8048298667110481
Test AP score: 0.8059014199486922
Epoch: 0017 train_loss= 1.22451 val_ap= 0.80013 time= 0.41915
Test AUC score: 0.8060519333358188
Test AP score: 0.8071623548105082
Epoch: 0018 train_loss= 1.22228 val_ap= 0.80325 time= 0.40920
Test AUC score: 0.8084361709316852
Test AP score: 0.8095047075593875
Epoch: 0019 train_loss= 1.22012 val_ap= 0.80662 time= 0.45201
Test AUC score: 0.8110962553736397
Test AP score: 0.8121902652080356
Epoch: 0020 train_loss= 1.21791 val_ap= 0.80954 time= 0.39825
Test AUC score: 0.8134905015718863
Test AP score: 0.8145121831472062
Epoch: 0021 train_loss= 1.21569 val_ap= 0.81204 time= 0.39825
Test AUC score: 0.815465600250535
Test AP score: 0.8163422093950791
Epoch: 0022 train_loss= 1.21350 val_ap= 0.81434 time= 0.39526
Test AUC score: 0.8175368452524554
Test AP score: 0.818364950744485
Epoch: 0023 train_loss= 1.21140 val_ap= 0.81682 time= 0.38729
Test AUC score: 0.8200505233705682
Test AP score: 0.8208171681560974
Epoch: 0024 train_loss= 1.20954 val_ap= 0.81878 time= 0.43210
Test AUC score: 0.8222171281118329
Test AP score: 0.8230745192481798
Epoch: 0025 train_loss= 1.20794 val_ap= 0.81978 time= 0.60434
Test AUC score: 0.8235789557824422
Test AP score: 0.8242637100660937
Epoch: 0026 train_loss= 1.20659 val_ap= 0.81959 time= 0.37734
Test AUC score: 0.8239558651013447
Test AP score: 0.8244312245210703
Epoch: 0027 train_loss= 1.20539 val_ap= 0.81884 time= 0.43807
Test AUC score: 0.8239585028698582
Test AP score: 0.8241554731081782
Epoch: 0028 train_loss= 1.20412 val_ap= 0.81872 time= 0.39426
Test AUC score: 0.8238559079145654
Test AP score: 0.8240186333400397
Epoch: 0029 train_loss= 1.20277 val_ap= 0.81949 time= 0.42612
Test AUC score: 0.8242985648003232
Test AP score: 0.824372131552863
Epoch: 0030 train_loss= 1.20168 val_ap= 0.82031 time= 0.42811
Test AUC score: 0.8245815315871289
Test AP score: 0.8246444744433723
Epoch: 0031 train_loss= 1.20091 val_ap= 0.82054 time= 0.38928
Test AUC score: 0.8239754347669743
Test AP score: 0.8240563888875255
Epoch: 0032 train_loss= 1.20014 val_ap= 0.82002 time= 0.38928
Test AUC score: 0.8228036145809822
Test AP score: 0.8230100758472507
Epoch: 0033 train_loss= 1.19913 val_ap= 0.81988 time= 0.38331
Test AUC score: 0.8223725123418748
Test AP score: 0.822707417451426
Epoch: 0034 train_loss= 1.19801 val_ap= 0.82005 time= 0.42513
Test AUC score: 0.8228458392198855
Test AP score: 0.8231310539788965
Epoch: 0035 train_loss= 1.19703 val_ap= 0.82018 time= 0.40024
Test AUC score: 0.8232131467901632
Test AP score: 0.8235392391918127
Epoch: 0036 train_loss= 1.19609 val_ap= 0.82013 time= 0.42413
Test AUC score: 0.8232227960050595
Test AP score: 0.8237980038052422
Epoch: 0037 train_loss= 1.19496 val_ap= 0.82096 time= 0.42911
Test AUC score: 0.8236873416228521
Test AP score: 0.8244946059868565
Epoch: 0038 train_loss= 1.19362 val_ap= 0.82282 time= 0.41517
Test AUC score: 0.8247046455334803
Test AP score: 0.8257257337040048
Epoch: 0039 train_loss= 1.19231 val_ap= 0.82517 time= 0.37535
Test AUC score: 0.8260484292400423
Test AP score: 0.827253612375422
Epoch: 0040 train_loss= 1.19106 val_ap= 0.82710 time= 0.40920
Test AUC score: 0.8271797946009666
Test AP score: 0.8285699895059979
Epoch: 0041 train_loss= 1.18981 val_ap= 0.82865 time= 0.39128
Test AUC score: 0.8284079504702748
Test AP score: 0.8298917438975075
Epoch: 0042 train_loss= 1.18859 val_ap= 0.83008 time= 0.40223
Test AUC score: 0.8298893456786687
Test AP score: 0.831270754556019
Epoch: 0043 train_loss= 1.18751 val_ap= 0.83165 time= 0.41517
Test AUC score: 0.8315058977045703
Test AP score: 0.8326198408670946
Epoch: 0044 train_loss= 1.18657 val_ap= 0.83259 time= 0.41318
Test AUC score: 0.8327920641384874
Test AP score: 0.8335660440889913
Epoch: 0045 train_loss= 1.18568 val_ap= 0.83327 time= 0.38032
Test AUC score: 0.8338737051896189
Test AP score: 0.8341778740922419
Epoch: 0046 train_loss= 1.18482 val_ap= 0.83403 time= 0.39924
Test AUC score: 0.8348295606214062
Test AP score: 0.8347976260793778
Epoch: 0047 train_loss= 1.18396 val_ap= 0.83510 time= 0.34349
Test AUC score: 0.835625651364386
Test AP score: 0.8355286024814521
Epoch: 0048 train_loss= 1.18303 val_ap= 0.83618 time= 0.35643
Test AUC score: 0.8363461147751531
Test AP score: 0.83633796021105
Epoch: 0049 train_loss= 1.18213 val_ap= 0.83701 time= 0.41119
Test AUC score: 0.8369188495873737
Test AP score: 0.8369828318098889
Epoch: 0050 train_loss= 1.18132 val_ap= 0.83748 time= 0.41617
Test AUC score: 0.8374896247206729
Test AP score: 0.837471628583642
Epoch: 0051 train_loss= 1.18054 val_ap= 0.83788 time= 0.42612
Test AUC score: 0.8380922904074909
Test AP score: 0.8379299745145476
Epoch: 0052 train_loss= 1.17977 val_ap= 0.83833 time= 0.40820
Test AUC score: 0.8386539520166718
Test AP score: 0.8384041174236634
Epoch: 0053 train_loss= 1.17910 val_ap= 0.83847 time= 0.36838
Test AUC score: 0.8390076299861471
Test AP score: 0.838733103819062
Epoch: 0054 train_loss= 1.17849 val_ap= 0.83863 time= 0.40720
Test AUC score: 0.8390638842987123
Test AP score: 0.8389363606861099
Epoch: 0055 train_loss= 1.17790 val_ap= 0.83880 time= 0.39924
Test AUC score: 0.8394093166987469
Test AP score: 0.8392843671995525
Epoch: 0056 train_loss= 1.17731 val_ap= 0.83922 time= 0.48287
Test AUC score: 0.8398248703435062
Test AP score: 0.8396962335475331
Epoch: 0057 train_loss= 1.17671 val_ap= 0.84004 time= 0.39426
Test AUC score: 0.8403976594028941
Test AP score: 0.8402131489905906
Epoch: 0058 train_loss= 1.17614 val_ap= 0.84071 time= 0.41119
Test AUC score: 0.8410275368250454
Test AP score: 0.8406722546630803
Epoch: 0059 train_loss= 1.17559 val_ap= 0.84107 time= 0.41915
Test AUC score: 0.8412319740561837
Test AP score: 0.8408530975661058
Epoch: 0060 train_loss= 1.17500 val_ap= 0.84136 time= 0.39924
Test AUC score: 0.8415677097750507
Test AP score: 0.8412051810833574
Epoch: 0061 train_loss= 1.17439 val_ap= 0.84155 time= 0.45898
Test AUC score: 0.841939709725306
Test AP score: 0.8415705023104798
Epoch: 0062 train_loss= 1.17380 val_ap= 0.84159 time= 0.43508
Test AUC score: 0.8423496013219709
Test AP score: 0.8419493625689581
Epoch: 0063 train_loss= 1.17321 val_ap= 0.84189 time= 0.41417
Test AUC score: 0.8427721867558009
Test AP score: 0.8423899044812054
Epoch: 0064 train_loss= 1.17263 val_ap= 0.84236 time= 0.42513
Test AUC score: 0.843214124866591
Test AP score: 0.8428699978782898
Epoch: 0065 train_loss= 1.17209 val_ap= 0.84307 time= 0.43210
Test AUC score: 0.8437657100644322
Test AP score: 0.8434961695348968
Epoch: 0066 train_loss= 1.17159 val_ap= 0.84343 time= 0.40322
Test AUC score: 0.8440212142227558
Test AP score: 0.8438664909832495
Epoch: 0067 train_loss= 1.17107 val_ap= 0.84340 time= 0.43608
Test AUC score: 0.8439417963697224
Test AP score: 0.8438890932128065
Epoch: 0068 train_loss= 1.17055 val_ap= 0.84337 time= 0.43110
Test AUC score: 0.8439967690929585
Test AP score: 0.8439480604270724
Epoch: 0069 train_loss= 1.17000 val_ap= 0.84371 time= 0.40322
Test AUC score: 0.8443325115927214
Test AP score: 0.8442758479791064
Epoch: 0070 train_loss= 1.16947 val_ap= 0.84408 time= 0.39924
Test AUC score: 0.8447661295442157
Test AP score: 0.8446396575209331
Epoch: 0071 train_loss= 1.16894 val_ap= 0.84429 time= 0.41517
Test AUC score: 0.8448746170980643
Test AP score: 0.8447030353399104
Epoch: 0072 train_loss= 1.16843 val_ap= 0.84445 time= 0.41318
Test AUC score: 0.8446942045811761
Test AP score: 0.8446207525066852
Epoch: 0073 train_loss= 1.16794 val_ap= 0.84479 time= 0.41915
Test AUC score: 0.8450775828747877
Test AP score: 0.8448549218902145
Epoch: 0074 train_loss= 1.16746 val_ap= 0.84513 time= 0.44006
Test AUC score: 0.8454585946357227
Test AP score: 0.8451423995288537
Epoch: 0075 train_loss= 1.16699 val_ap= 0.84540 time= 0.40820
Test AUC score: 0.8457033781975943
Test AP score: 0.8453491159382689
Epoch: 0076 train_loss= 1.16654 val_ap= 0.84549 time= 0.39227
Test AUC score: 0.845770468381842
Test AP score: 0.8454660888158692
Epoch: 0077 train_loss= 1.16612 val_ap= 0.84562 time= 0.42214
Test AUC score: 0.8457238429414854
Test AP score: 0.8454943048426693
Epoch: 0078 train_loss= 1.16572 val_ap= 0.84589 time= 0.41019
Test AUC score: 0.8459026280433525
Test AP score: 0.8456890742831722
Epoch: 0079 train_loss= 1.16535 val_ap= 0.84624 time= 0.37236
Test AUC score: 0.846082681172757
Test AP score: 0.8458027257571787
Epoch: 0080 train_loss= 1.16500 val_ap= 0.84631 time= 0.39227
Test AUC score: 0.8462803917551414
Test AP score: 0.8458901813884252
Epoch: 0081 train_loss= 1.16469 val_ap= 0.84649 time= 0.42811
Test AUC score: 0.8464165792688293
Test AP score: 0.8459844219622428
Epoch: 0082 train_loss= 1.16439 val_ap= 0.84681 time= 0.39825
Test AUC score: 0.8468448877788171
Test AP score: 0.8462796249337834
Epoch: 0083 train_loss= 1.16411 val_ap= 0.84689 time= 0.42015
Test AUC score: 0.8469719821110744
Test AP score: 0.8464016494590783
Epoch: 0084 train_loss= 1.16384 val_ap= 0.84671 time= 0.40422
Test AUC score: 0.846961898918839
Test AP score: 0.8463815814209839
Epoch: 0085 train_loss= 1.16357 val_ap= 0.84663 time= 0.40721
Test AUC score: 0.8469176603538469
Test AP score: 0.8463287137444744
Epoch: 0086 train_loss= 1.16330 val_ap= 0.84679 time= 0.41418
Test AUC score: 0.8469351211608448
Test AP score: 0.8463626005803428
Epoch: 0087 train_loss= 1.16303 val_ap= 0.84698 time= 0.40920
Test AUC score: 0.8470235643863494
Test AP score: 0.8464284506056875
Epoch: 0088 train_loss= 1.16277 val_ap= 0.84701 time= 0.44703
Test AUC score: 0.8467976520578275
Test AP score: 0.8463391004778882
Epoch: 0089 train_loss= 1.16251 val_ap= 0.84696 time= 0.42612
Test AUC score: 0.8468884211306332
Test AP score: 0.8464271943540462
Epoch: 0090 train_loss= 1.16225 val_ap= 0.84701 time= 0.46197
Test AUC score: 0.8470199230452397
Test AP score: 0.8465434759967414
Epoch: 0091 train_loss= 1.16200 val_ap= 0.84700 time= 0.44006
Test AUC score: 0.8471591348385088
Test AP score: 0.8466382520644085
Epoch: 0092 train_loss= 1.16175 val_ap= 0.84700 time= 0.40820
Test AUC score: 0.8472114494505433
Test AP score: 0.8466988541241995
Epoch: 0093 train_loss= 1.16151 val_ap= 0.84714 time= 0.37335
Test AUC score: 0.8471928697957184
Test AP score: 0.8467521351776309
Epoch: 0094 train_loss= 1.16126 val_ap= 0.84730 time= 0.41418
Test AUC score: 0.8472978923117509
Test AP score: 0.8468748849571877
Epoch: 0095 train_loss= 1.16102 val_ap= 0.84764 time= 0.41617
Test AUC score: 0.8476720146824298
Test AP score: 0.8471587491660437
Epoch: 0096 train_loss= 1.16078 val_ap= 0.84774 time= 0.37136
Test AUC score: 0.8478381669751952
Test AP score: 0.8473041063945792
Epoch: 0097 train_loss= 1.16054 val_ap= 0.84787 time= 0.41517
Test AUC score: 0.8479585617822821
Test AP score: 0.84741566927706
Epoch: 0098 train_loss= 1.16032 val_ap= 0.84810 time= 0.43708
Test AUC score: 0.8480852967270556
Test AP score: 0.847584690740093
Epoch: 0099 train_loss= 1.16009 val_ap= 0.84817 time= 0.40820
Test AUC score: 0.8484615550799496
Test AP score: 0.8478526043668483
Epoch: 0100 train_loss= 1.15986 val_ap= 0.84842 time= 0.45997
Test AUC score: 0.8487554933563562
Test AP score: 0.8481096899040907
Epoch: 0101 train_loss= 1.15964 val_ap= 0.84852 time= 0.40721
Test AUC score: 0.8490919817546705
Test AP score: 0.8483760885406337
Epoch: 0102 train_loss= 1.15941 val_ap= 0.84874 time= 0.44504
Test AUC score: 0.8492506954046098
Test AP score: 0.8485743149257101
Epoch: 0103 train_loss= 1.15919 val_ap= 0.84882 time= 0.41019
Test AUC score: 0.8494621101776516
Test AP score: 0.8487712621773155
Epoch: 0104 train_loss= 1.15896 val_ap= 0.84890 time= 0.40820
Test AUC score: 0.8497501219510228
Test AP score: 0.8489801619943376
Epoch: 0105 train_loss= 1.15875 val_ap= 0.84909 time= 0.44006
Test AUC score: 0.8500799243859584
Test AP score: 0.849222165504127
Epoch: 0106 train_loss= 1.15853 val_ap= 0.84930 time= 0.39426
Test AUC score: 0.8503218735333348
Test AP score: 0.8494440316371498
Epoch: 0107 train_loss= 1.15832 val_ap= 0.84931 time= 0.40820
Test AUC score: 0.8503874447968958
Test AP score: 0.8495095002681378
Epoch: 0108 train_loss= 1.15811 val_ap= 0.84938 time= 0.41617
Test AUC score: 0.8505945604819232
Test AP score: 0.8496470273890194
Epoch: 0109 train_loss= 1.15790 val_ap= 0.84968 time= 0.39280
Test AUC score: 0.850788670408573
Test AP score: 0.8498485551371022
Epoch: 0110 train_loss= 1.15769 val_ap= 0.84977 time= 0.39625
Test AUC score: 0.850818058811497
Test AP score: 0.8498789565442996
Epoch: 0111 train_loss= 1.15748 val_ap= 0.84984 time= 0.40522
Test AUC score: 0.8509289332407098
Test AP score: 0.8499999575792713
Epoch: 0112 train_loss= 1.15727 val_ap= 0.85000 time= 0.40322
Test AUC score: 0.8511069181968584
Test AP score: 0.8501701684657674
Epoch: 0113 train_loss= 1.15706 val_ap= 0.85018 time= 0.40322
Test AUC score: 0.8511577952589577
Test AP score: 0.8502310538521862
Epoch: 0114 train_loss= 1.15685 val_ap= 0.85031 time= 0.40322
Test AUC score: 0.8511930355750619
Test AP score: 0.8502935849080226
Epoch: 0115 train_loss= 1.15664 val_ap= 0.85050 time= 0.41418
Test AUC score: 0.8513740448107913
Test AP score: 0.8504587883878084
Epoch: 0116 train_loss= 1.15643 val_ap= 0.85057 time= 0.41716
Test AUC score: 0.8514962704597769
Test AP score: 0.8505409146232524
Epoch: 0117 train_loss= 1.15622 val_ap= 0.85055 time= 0.41417
Test AUC score: 0.8515250011157964
Test AP score: 0.8505542981000909
Epoch: 0118 train_loss= 1.15601 val_ap= 0.85077 time= 0.34648
Test AUC score: 0.8516248837127205
Test AP score: 0.8506712052884524
Epoch: 0119 train_loss= 1.15580 val_ap= 0.85084 time= 0.40920
Test AUC score: 0.8517709984580378
Test AP score: 0.8507818477932206
Epoch: 0120 train_loss= 1.15559 val_ap= 0.85095 time= 0.42114
Test AUC score: 0.8518342913405695
Test AP score: 0.8508423275110107
Epoch: 0121 train_loss= 1.15538 val_ap= 0.85114 time= 0.43409
Test AUC score: 0.8519995892404486
Test AP score: 0.8510231668520386
Epoch: 0122 train_loss= 1.15516 val_ap= 0.85122 time= 0.36838
Test AUC score: 0.852105771289684
Test AP score: 0.8511045408719929
Epoch: 0123 train_loss= 1.15495 val_ap= 0.85141 time= 0.49183
Test AUC score: 0.852308852341638
Test AP score: 0.8512782889738562
Epoch: 0124 train_loss= 1.15473 val_ap= 0.85159 time= 0.40721
Test AUC score: 0.8524459010291081
Test AP score: 0.8514032045706722
Epoch: 0125 train_loss= 1.15451 val_ap= 0.85171 time= 0.42214
Test AUC score: 0.852668436471461
Test AP score: 0.8515647866267337
Epoch: 0126 train_loss= 1.15428 val_ap= 0.85181 time= 0.40422
Test AUC score: 0.852836684061066
Test AP score: 0.8517179253568595
Epoch: 0127 train_loss= 1.15405 val_ap= 0.85182 time= 0.41218
Test AUC score: 0.8529751160513043
Test AP score: 0.851829793100243
Epoch: 0128 train_loss= 1.15381 val_ap= 0.85193 time= 0.37833
Test AUC score: 0.8531266216088789
Test AP score: 0.8519735185516706
Epoch: 0129 train_loss= 1.15357 val_ap= 0.85207 time= 0.39327
Test AUC score: 0.8532283757330776
Test AP score: 0.8521188829879734
Epoch: 0130 train_loss= 1.15332 val_ap= 0.85214 time= 0.38431
Test AUC score: 0.8532627006282324
Test AP score: 0.8521716555754119
Epoch: 0131 train_loss= 1.15307 val_ap= 0.85232 time= 0.41617
Test AUC score: 0.8534344810646126
Test AP score: 0.8523259131278488
Epoch: 0132 train_loss= 1.15281 val_ap= 0.85235 time= 0.38032
Test AUC score: 0.8535426702590406
Test AP score: 0.8524430493508627
Epoch: 0133 train_loss= 1.15254 val_ap= 0.85244 time= 0.40920
Test AUC score: 0.8536604069549263
Test AP score: 0.8526133860253615
Epoch: 0134 train_loss= 1.15227 val_ap= 0.85248 time= 0.44205
Test AUC score: 0.8536020370028337
Test AP score: 0.8526591098772247
Epoch: 0135 train_loss= 1.15199 val_ap= 0.85255 time= 0.43010
Test AUC score: 0.8537281684478701
Test AP score: 0.8527857128852345
Epoch: 0136 train_loss= 1.15171 val_ap= 0.85270 time= 0.41019
Test AUC score: 0.853894971706644
Test AP score: 0.8529336209626202
Epoch: 0137 train_loss= 1.15142 val_ap= 0.85286 time= 0.44305
Test AUC score: 0.8540717428824225
Test AP score: 0.853110603469259
Epoch: 0138 train_loss= 1.15113 val_ap= 0.85286 time= 0.44604
Test AUC score: 0.8541696522386335
Test AP score: 0.8532065104291895
Epoch: 0139 train_loss= 1.15084 val_ap= 0.85301 time= 0.39626
Test AUC score: 0.8543609209698921
Test AP score: 0.8534005131007061
Epoch: 0140 train_loss= 1.15054 val_ap= 0.85317 time= 0.42911
Test AUC score: 0.8543868985821675
Test AP score: 0.8534374488190641
Epoch: 0141 train_loss= 1.15025 val_ap= 0.85328 time= 0.46197
Test AUC score: 0.8544227152744251
Test AP score: 0.8534839157303331
Epoch: 0142 train_loss= 1.14997 val_ap= 0.85345 time= 0.39426
Test AUC score: 0.8545556683007587
Test AP score: 0.8536272351113947
Epoch: 0143 train_loss= 1.14968 val_ap= 0.85341 time= 0.40621
Test AUC score: 0.8545699692102571
Test AP score: 0.8536580835513602
Epoch: 0144 train_loss= 1.14940 val_ap= 0.85358 time= 0.41617
Test AUC score: 0.8547959493477382
Test AP score: 0.8538464794246146
Epoch: 0145 train_loss= 1.14911 val_ap= 0.85353 time= 0.45400
Test AUC score: 0.8548085889377359
Test AP score: 0.8538340154847015
Epoch: 0146 train_loss= 1.14883 val_ap= 0.85364 time= 0.50677
Test AUC score: 0.8550939086954262
Test AP score: 0.8540485516462919
Epoch: 0147 train_loss= 1.14855 val_ap= 0.85359 time= 0.37734
Test AUC score: 0.8550020004321057
Test AP score: 0.8540379841784539
Epoch: 0148 train_loss= 1.14829 val_ap= 0.85367 time= 0.37833
Test AUC score: 0.8551682137529344
Test AP score: 0.8541923491504169
Epoch: 0149 train_loss= 1.14804 val_ap= 0.85375 time= 0.40820
Test AUC score: 0.8553570006762858
Test AP score: 0.8543496079957483
Epoch: 0150 train_loss= 1.14781 val_ap= 0.85369 time= 0.45002
Test AUC score: 0.8554939612121091
Test AP score: 0.8544154566105616
Epoch: 0151 train_loss= 1.14759 val_ap= 0.85381 time= 0.40820
Test AUC score: 0.8556614764649546
Test AP score: 0.8545688655143662
Epoch: 0152 train_loss= 1.14738 val_ap= 0.85383 time= 0.39725
Test AUC score: 0.8556762113517921
Test AP score: 0.854565570903513
Epoch: 0153 train_loss= 1.14718 val_ap= 0.85404 time= 0.39825
Test AUC score: 0.8558668358979381
Test AP score: 0.8547229629023955
Epoch: 0154 train_loss= 1.14699 val_ap= 0.85421 time= 0.44504
Test AUC score: 0.8560921582885148
Test AP score: 0.8549073423437313
Epoch: 0155 train_loss= 1.14679 val_ap= 0.85423 time= 0.47590
Test AUC score: 0.8562962632557529
Test AP score: 0.8550699154364507
Epoch: 0156 train_loss= 1.14659 val_ap= 0.85443 time= 0.39825
Test AUC score: 0.8564575943315181
Test AP score: 0.8551829406477971
Epoch: 0157 train_loss= 1.14640 val_ap= 0.85453 time= 0.38132
Test AUC score: 0.8566497242365589
Test AP score: 0.855292965298877
Epoch: 0158 train_loss= 1.14620 val_ap= 0.85467 time= 0.38829
Test AUC score: 0.8567913500287763
Test AP score: 0.8554172255102704
Epoch: 0159 train_loss= 1.14601 val_ap= 0.85494 time= 0.40621
Test AUC score: 0.8569232181117622
Test AP score: 0.8555696218950701
Epoch: 0160 train_loss= 1.14581 val_ap= 0.85504 time= 0.40820
Test AUC score: 0.8569426182549937
Test AP score: 0.855587851803751
Epoch: 0161 train_loss= 1.14562 val_ap= 0.85524 time= 0.43011
Test AUC score: 0.8569482124941288
Test AP score: 0.85566930300085
Epoch: 0162 train_loss= 1.14544 val_ap= 0.85527 time= 0.41915
Test AUC score: 0.8569173661985819
Test AP score: 0.8556355877925832
Epoch: 0163 train_loss= 1.14526 val_ap= 0.85555 time= 0.39327
Test AUC score: 0.8571192131274782
Test AP score: 0.8558183429202006
Epoch: 0164 train_loss= 1.14509 val_ap= 0.85542 time= 0.40920
Test AUC score: 0.8569842868604314
Test AP score: 0.8556511939679735
Epoch: 0165 train_loss= 1.14492 val_ap= 0.85605 time= 0.40322
Test AUC score: 0.8575442057793603
Test AP score: 0.8562034008288104
Epoch: 0166 train_loss= 1.14480 val_ap= 0.85508 time= 0.43707
Test AUC score: 0.8565860855283355
Test AP score: 0.855363993982124
Epoch: 0167 train_loss= 1.14470 val_ap= 0.85660 time= 0.36738
Test AUC score: 0.8580547597759803
Test AP score: 0.8567287743752965
Epoch: 0168 train_loss= 1.14467 val_ap= 0.85469 time= 0.45300
Test AUC score: 0.8559379471534674
Test AP score: 0.8547921544741413
Epoch: 0169 train_loss= 1.14446 val_ap= 0.85701 time= 0.38132
Test AUC score: 0.8584922292763582
Test AP score: 0.8571185617134706
Epoch: 0170 train_loss= 1.14420 val_ap= 0.85514 time= 0.39526
Test AUC score: 0.8566255978088702
Test AP score: 0.8554041896681515
Epoch: 0171 train_loss= 1.14392 val_ap= 0.85640 time= 0.41417
Test AUC score: 0.8577935393223951
Test AP score: 0.8564699471227196
Epoch: 0172 train_loss= 1.14377 val_ap= 0.85666 time= 0.43409
Test AUC score: 0.858067575669272
Test AP score: 0.8566970642515479
Epoch: 0173 train_loss= 1.14370 val_ap= 0.85579 time= 0.40920
Test AUC score: 0.8571857065928847
Test AP score: 0.8558788067317574
Epoch: 0174 train_loss= 1.14357 val_ap= 0.85723 time= 0.46993
Test AUC score: 0.8587921415220716
Test AP score: 0.8573676332390263
Epoch: 0175 train_loss= 1.14339 val_ap= 0.85593 time= 0.39825
Test AUC score: 0.8573534049299202
Test AP score: 0.8559879613274459
Epoch: 0176 train_loss= 1.14316 val_ap= 0.85703 time= 0.46495
Test AUC score: 0.858633868630367
Test AP score: 0.8571906381269788
Epoch: 0177 train_loss= 1.14299 val_ap= 0.85685 time= 0.43110
Test AUC score: 0.8584676349668507
Test AP score: 0.8570751013542957
Epoch: 0178 train_loss= 1.14288 val_ap= 0.85649 time= 0.44006
Test AUC score: 0.8579145986572823
Test AP score: 0.8565590991446141
Epoch: 0179 train_loss= 1.14277 val_ap= 0.85749 time= 0.40820
Test AUC score: 0.8590499783085921
Test AP score: 0.8576207139706663
Epoch: 0180 train_loss= 1.14265 val_ap= 0.85632 time= 0.46396
Test AUC score: 0.8577772380485997
Test AP score: 0.8564908886109834
Epoch: 0181 train_loss= 1.14247 val_ap= 0.85762 time= 0.41119
Test AUC score: 0.8592283972420794
Test AP score: 0.8578231486693726
Epoch: 0182 train_loss= 1.14228 val_ap= 0.85685 time= 0.41119
Test AUC score: 0.8582403122119772
Test AP score: 0.8569286647744992
Epoch: 0183 train_loss= 1.14212 val_ap= 0.85738 time= 0.40322
Test AUC score: 0.8587337919126667
Test AP score: 0.8574151526265484
Epoch: 0184 train_loss= 1.14198 val_ap= 0.85746 time= 0.38530
Test AUC score: 0.8589468747861051
Test AP score: 0.8575960628617074
Epoch: 0185 train_loss= 1.14186 val_ap= 0.85713 time= 0.41816
Test AUC score: 0.8586025273294178
Test AP score: 0.8572875285483994
Epoch: 0186 train_loss= 1.14174 val_ap= 0.85795 time= 0.44205
Test AUC score: 0.8595579962501373
Test AP score: 0.858129612399972
Epoch: 0187 train_loss= 1.14162 val_ap= 0.85713 time= 0.39426
Test AUC score: 0.8586028392506302
Test AP score: 0.8572401510435774
Epoch: 0188 train_loss= 1.14149 val_ap= 0.85828 time= 0.38730
Test AUC score: 0.8598099336572061
Test AP score: 0.8584224781461133
Epoch: 0189 train_loss= 1.14136 val_ap= 0.85734 time= 0.42612
Test AUC score: 0.8586698616259185
Test AP score: 0.8573600246187421
Epoch: 0190 train_loss= 1.14119 val_ap= 0.85838 time= 0.43508
Test AUC score: 0.8599621579897487
Test AP score: 0.8585396302640035
Epoch: 0191 train_loss= 1.14104 val_ap= 0.85766 time= 0.41517
Test AUC score: 0.8589819794842909
Test AP score: 0.8576438392470374
Epoch: 0192 train_loss= 1.14088 val_ap= 0.85833 time= 0.39526
Test AUC score: 0.859870609113912
Test AP score: 0.8585330321424558
Epoch: 0193 train_loss= 1.14072 val_ap= 0.85803 time= 0.41318
Test AUC score: 0.8593654323677578
Test AP score: 0.858054940667632
Epoch: 0194 train_loss= 1.14058 val_ap= 0.85839 time= 0.42214
Test AUC score: 0.8597094679032328
Test AP score: 0.8583838022789818
Epoch: 0195 train_loss= 1.14044 val_ap= 0.85841 time= 0.41218
Test AUC score: 0.8597163640743847
Test AP score: 0.8584308601101808
Epoch: 0196 train_loss= 1.14030 val_ap= 0.85835 time= 0.44305
Test AUC score: 0.8596018686467497
Test AP score: 0.8583876927798523
Epoch: 0197 train_loss= 1.14016 val_ap= 0.85871 time= 0.42114
Test AUC score: 0.8599299962003928
Test AP score: 0.8586749427222324
Epoch: 0198 train_loss= 1.14004 val_ap= 0.85828 time= 0.41318
Test AUC score: 0.8595628852760969
Test AP score: 0.8583337187957087
Epoch: 0199 train_loss= 1.13992 val_ap= 0.85901 time= 0.42513
Test AUC score: 0.8601945257311877
Test AP score: 0.859021918841075
Epoch: 0200 train_loss= 1.13982 val_ap= 0.85813 time= 0.41019
Test AUC score: 0.8593704298880518
Test AP score: 0.85819134453468
Epoch: 0001 train_loss= 1.13975 val_ap= 0.85961 time= 0.42347
Test AUC score: 0.8606795360928714
Test AP score: 0.8595159304299275
Epoch  0  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16665278375148773
Epoch: 0002 train_loss= 1.13977 val_ap= 0.85857 time= 0.39825
Test AUC score: 0.8594024899639688
Test AP score: 0.8585466751597763
Epoch  1  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16694064438343048
Epoch: 0003 train_loss= 1.13992 val_ap= 0.86010 time= 0.45101
Test AUC score: 0.8606377386504112
Test AP score: 0.859951326604361
Epoch  2  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1671377271413803
Epoch: 0004 train_loss= 1.13957 val_ap= 0.85898 time= 0.44457
Test AUC score: 0.8591840908681282
Test AP score: 0.858235124927913
Epoch  3  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16706982254981995
Epoch: 0005 train_loss= 1.13979 val_ap= 0.86114 time= 0.39526
Test AUC score: 0.8617269607431829
Test AP score: 0.8604651100808092
Epoch  4  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16739659011363983
Epoch: 0006 train_loss= 1.14004 val_ap= 0.85829 time= 0.45599
Test AUC score: 0.8583673997633386
Test AP score: 0.8576936091462412
Epoch  5  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16738060116767883
Epoch: 0007 train_loss= 1.14018 val_ap= 0.86196 time= 0.39227
Test AUC score: 0.8617690836686475
Test AP score: 0.8612017194265715
Epoch  6  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16810382902622223
Epoch: 0008 train_loss= 1.13977 val_ap= 0.85852 time= 0.43110
Test AUC score: 0.8585986893423262
Test AP score: 0.858038767205183
Epoch  7  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16804037988185883
Epoch: 0009 train_loss= 1.13942 val_ap= 0.86078 time= 0.45599
Test AUC score: 0.8616457527336266
Test AP score: 0.8606233127343408
Epoch  8  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1685030311346054
Epoch: 0010 train_loss= 1.13919 val_ap= 0.85994 time= 0.39626
Test AUC score: 0.8604938616007478
Test AP score: 0.8597937757253563
Epoch  9  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16822439432144165
Epoch: 0011 train_loss= 1.13927 val_ap= 0.85948 time= 0.43010
Test AUC score: 0.8591437987845623
Test AP score: 0.8586558454486815
Epoch  10  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16861216723918915
Epoch: 0012 train_loss= 1.13930 val_ap= 0.86105 time= 0.45201
Test AUC score: 0.8614015794523882
Test AP score: 0.8606310187860834
Epoch  11  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16948026418685913
Epoch: 0013 train_loss= 1.13938 val_ap= 0.85829 time= 0.39625
Test AUC score: 0.8586806229077462
Test AP score: 0.8582312905479716
Epoch  12  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16920730471611023
Epoch: 0014 train_loss= 1.13935 val_ap= 0.86075 time= 0.45201
Test AUC score: 0.8613038328376792
Test AP score: 0.8604448140363552
Epoch  13  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1693611592054367
Epoch: 0015 train_loss= 1.13911 val_ap= 0.85984 time= 0.41218
Test AUC score: 0.8598968104957527
Test AP score: 0.8590999080156797
Epoch  14  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16926614940166473
Epoch: 0016 train_loss= 1.13904 val_ap= 0.85975 time= 0.41219
Test AUC score: 0.8599042423576827
Test AP score: 0.8592475961706106
Epoch  15  : 
Ranking optimizing... 
Now Average NDCG@k =  0.16937527060508728
Epoch: 0017 train_loss= 1.13933 val_ap= 0.85996 time= 0.41418
Test AUC score: 0.8604868298116773
Test AP score: 0.8597761343627228
Epoch  16  : 
Ranking optimizing... 
Now Average NDCG@k =  0.170089989900589
Epoch: 0018 train_loss= 1.13927 val_ap= 0.85834 time= 0.38829
Test AUC score: 0.8586380727858385
Test AP score: 0.8580795370469045
Epoch  17  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1701144129037857
Epoch: 0019 train_loss= 1.13942 val_ap= 0.85988 time= 0.38630
Test AUC score: 0.8600108855078409
Test AP score: 0.859442514323902
Epoch  18  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1706823706626892
Epoch: 0020 train_loss= 1.13906 val_ap= 0.85881 time= 0.44404
Test AUC score: 0.8587044645378062
Test AP score: 0.8583279801292771
Epoch  19  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17032742500305176
Epoch: 0021 train_loss= 1.13903 val_ap= 0.85969 time= 0.37634
Test AUC score: 0.8601840628087807
Test AP score: 0.8596613912925062
Epoch  20  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17047935724258423
Epoch: 0022 train_loss= 1.13898 val_ap= 0.86002 time= 0.37037
Test AUC score: 0.8605773886767094
Test AP score: 0.8599227142233671
Epoch  21  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17063869535923004
Epoch: 0023 train_loss= 1.13903 val_ap= 0.85929 time= 0.35245
Test AUC score: 0.8596198312400459
Test AP score: 0.8588072999970632
Epoch  22  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17031517624855042
Epoch: 0024 train_loss= 1.13912 val_ap= 0.86037 time= 0.37634
Test AUC score: 0.8609803569786384
Test AP score: 0.8600979039336631
Epoch  23  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17081773281097412
Epoch: 0025 train_loss= 1.13874 val_ap= 0.85849 time= 0.38630
Test AUC score: 0.8591014317468447
Test AP score: 0.8585656989045034
Epoch  24  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17066407203674316
Epoch: 0026 train_loss= 1.13864 val_ap= 0.85968 time= 0.33552
Test AUC score: 0.8596152270117152
Test AP score: 0.8591146963627481
Epoch  25  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17086124420166016
Epoch: 0027 train_loss= 1.13868 val_ap= 0.85929 time= 0.33851
Test AUC score: 0.8589263693568383
Test AP score: 0.8583962089532884
Epoch  26  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17124085128307343
Epoch: 0028 train_loss= 1.13884 val_ap= 0.85849 time= 0.39327
Test AUC score: 0.8586284845990053
Test AP score: 0.8580133937000388
Epoch  27  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17121009528636932
Epoch: 0029 train_loss= 1.13884 val_ap= 0.86011 time= 0.36240
Test AUC score: 0.8606792919806182
Test AP score: 0.859960383565072
Epoch  28  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17181062698364258
Epoch: 0030 train_loss= 1.13912 val_ap= 0.85802 time= 0.38431
Test AUC score: 0.8581917406544913
Test AP score: 0.8576362848362576
Epoch  29  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1711861938238144
Epoch: 0031 train_loss= 1.13884 val_ap= 0.86062 time= 0.39227
Test AUC score: 0.8610425445751348
Test AP score: 0.8601980021614934
Epoch  30  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1718740612268448
Epoch: 0032 train_loss= 1.13872 val_ap= 0.85861 time= 0.41617
Test AUC score: 0.8595074107665627
Test AP score: 0.8587558015577642
Epoch  31  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1716601550579071
Epoch: 0033 train_loss= 1.13854 val_ap= 0.85908 time= 0.36340
Test AUC score: 0.8598475269441951
Test AP score: 0.8591035953342294
Epoch  32  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17206814885139465
Epoch: 0034 train_loss= 1.13842 val_ap= 0.85921 time= 0.37037
Test AUC score: 0.8598068212259781
Test AP score: 0.8592216955549713
Epoch  33  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17176972329616547
Epoch: 0035 train_loss= 1.13840 val_ap= 0.85850 time= 0.39327
Test AUC score: 0.858795477722865
Test AP score: 0.858492932808378
Epoch  34  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17192083597183228
Epoch: 0036 train_loss= 1.13837 val_ap= 0.85963 time= 0.36340
Test AUC score: 0.860665750531463
Test AP score: 0.8599454714643084
Epoch  35  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17223425209522247
Epoch: 0037 train_loss= 1.13851 val_ap= 0.85790 time= 0.38032
Test AUC score: 0.8589528826598917
Test AP score: 0.8581482498854092
Epoch  36  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17233143746852875
Epoch: 0038 train_loss= 1.13829 val_ap= 0.85932 time= 0.36639
Test AUC score: 0.8603840721148824
Test AP score: 0.8596304036181577
Epoch  37  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17243149876594543
Epoch: 0039 train_loss= 1.13822 val_ap= 0.85883 time= 0.41816
Test AUC score: 0.8594708617375457
Test AP score: 0.85907999273533
Epoch  38  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17222952842712402
Epoch: 0040 train_loss= 1.13829 val_ap= 0.85870 time= 0.42015
Test AUC score: 0.8595932569089295
Test AP score: 0.8589539071980197
Epoch  39  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1724315732717514
Epoch: 0041 train_loss= 1.13815 val_ap= 0.85925 time= 0.36937
Test AUC score: 0.8599102027651979
Test AP score: 0.8593909076883475
Epoch  40  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17287568747997284
Epoch: 0042 train_loss= 1.13834 val_ap= 0.85817 time= 0.38729
Test AUC score: 0.8590013118185632
Test AP score: 0.8583566331220698
Epoch  41  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17306500673294067
Epoch: 0043 train_loss= 1.13839 val_ap= 0.85978 time= 0.46694
Test AUC score: 0.8611794305211027
Test AP score: 0.8602542966964409
Epoch  42  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17356134951114655
Epoch: 0044 train_loss= 1.13855 val_ap= 0.85738 time= 0.37236
Test AUC score: 0.858559027882081
Test AP score: 0.8580182163254806
Epoch  43  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17362642288208008
Epoch: 0045 train_loss= 1.13823 val_ap= 0.85983 time= 0.36639
Test AUC score: 0.8608178799314624
Test AP score: 0.8601987436159706
Epoch  44  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17398524284362793
Epoch: 0046 train_loss= 1.13810 val_ap= 0.85843 time= 0.33851
Test AUC score: 0.859215737309394
Test AP score: 0.8587640152682415
Epoch  45  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17399407923221588
Epoch: 0047 train_loss= 1.13798 val_ap= 0.85842 time= 0.36539
Test AUC score: 0.8593401531877621
Test AP score: 0.8590231021897037
Epoch  46  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17412415146827698
Epoch: 0048 train_loss= 1.13802 val_ap= 0.85903 time= 0.39426
Test AUC score: 0.8602041206989167
Test AP score: 0.8597643339343238
Epoch  47  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1742801070213318
Epoch: 0049 train_loss= 1.13824 val_ap= 0.85774 time= 0.37535
Test AUC score: 0.8589085559432527
Test AP score: 0.8584235655132378
Epoch  48  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17415715754032135
Epoch: 0050 train_loss= 1.13816 val_ap= 0.85907 time= 0.37834
Test AUC score: 0.860414423405027
Test AP score: 0.8599996214996783
Epoch  49  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17472247779369354
Epoch: 0051 train_loss= 1.13837 val_ap= 0.85755 time= 0.37833
Test AUC score: 0.8584138624621929
Test AP score: 0.8582812357910597
Epoch  50  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17483992874622345
Epoch: 0052 train_loss= 1.13810 val_ap= 0.85927 time= 0.39426
Test AUC score: 0.86017220980271
Test AP score: 0.8599021694422833
Epoch  51  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17500406503677368
Epoch: 0053 train_loss= 1.13811 val_ap= 0.85829 time= 0.40223
Test AUC score: 0.8594838878386111
Test AP score: 0.859129142655868
Epoch  52  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17510661482810974
Epoch: 0054 train_loss= 1.13816 val_ap= 0.85843 time= 0.37535
Test AUC score: 0.859740809204182
Test AP score: 0.8592251994119647
Epoch  53  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1751708686351776
Epoch: 0055 train_loss= 1.13816 val_ap= 0.85802 time= 0.39426
Test AUC score: 0.8596173494321385
Test AP score: 0.8592785418971397
Epoch  54  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17545147240161896
Epoch: 0056 train_loss= 1.13817 val_ap= 0.85848 time= 0.30864
Test AUC score: 0.8593973839493398
Test AP score: 0.8591520863162633
Epoch  55  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17560157179832458
Epoch: 0057 train_loss= 1.13814 val_ap= 0.85890 time= 0.42214
Test AUC score: 0.8598557453900518
Test AP score: 0.8594654867428049
Epoch  56  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17582304775714874
Epoch: 0058 train_loss= 1.13837 val_ap= 0.85705 time= 0.37734
Test AUC score: 0.8586019645150563
Test AP score: 0.8584366154646237
Epoch  57  : 
Ranking optimizing... 
Now Average NDCG@k =  0.1759795844554901
Epoch: 0059 train_loss= 1.13838 val_ap= 0.85909 time= 0.38431
Test AUC score: 0.8608572904985585
Test AP score: 0.8604053043417057
Epoch  58  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17639575898647308
Epoch: 0060 train_loss= 1.13916 val_ap= 0.85671 time= 0.40522
Test AUC score: 0.8573534117108161
Test AP score: 0.8572467173867998
Epoch  59  : 
Ranking optimizing... 
Now Average NDCG@k =  0.17630824446678162
auc_before[0.8606795360928714]
auc_after[0.8573534117108161]
ap_before[0.8595159304299275]
ap_after[0.8572467173867998]
fair_before[0.16665278375148773]
fair_after[0.17630824446678162]

Process finished with exit code 0

```


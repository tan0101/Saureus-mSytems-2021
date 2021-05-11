# Saureus-mSytems-2021
"Whole-genome sequencing and machine learning analysis of Staphylococcus aureus from multiple heterogeneous sources in China reveals common genetic traits of antimicrobial resistance" by WEI WANG, Michelle Baker, Yue Hu, Jin Xu, Dajin Yang, Alexandre Guerra, Ning Xue, Hui Li, Shaofei Yan, Menghan Li, Yao Bai, Yinping Dong, Zixin Peng, Jinjing Ma, Fengqin Li, and Tania Dottorini accepted for publication in mSystems

Any questions should be made to the corresponding author Dr Tania Dottorini (Tania.Dottorini@nottingham.ac.uk)

Three scripts are available:

1. important_kmers.py -> find the top 2000 k-mers based on the pvalue of the Chi-square test from a k-mers dataset for each antibitioc studied.
2. classification_kmers.py -> measures the performance of 10 different classifiers using a nested cross-validation with the dataset acquired from important_kmers.py

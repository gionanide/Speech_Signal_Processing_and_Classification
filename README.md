# Speech-Signal-Processing-and-Classification

 #### University of Groningen
 
   Front-end speech processing aims at extracting proper features from short- term segments of a speech utterance, known as frames. It is a pre-requisite step toward any pattern recognition problem employing speech or audio (e.g., music). Here, we are interesting in voice disorder classification. 
   That is, to develop two-class classifiers, which can discriminate between utterances of a subject suffering from say vocal fold paralysis and utterances of a healthy subject.The mathematical modeling of the speech production system in humans suggests that an all-pole system function is justified [1-3]. As a consequence, linear prediction coefficients (LPCs) constitute a first choice for modeling the magnitute of the short-term spectrum of speech. LPC-derived cepstral coefficients are guaranteed to discriminate between the system (e.g., vocal tract) contribution and that of the excitation. Taking into account the characteristics of the human ear, the mel-frequency cepstral coefficients (MFCCs) emerged as descriptive features of the speech spectral envelope. Similarly to MFCCs, the perceptual linear prediction coefficients (PLPs) could also be derived. The aforementioned sort of speaking tradi- tional features will be tested against agnostic-features extracted by convolu- tive neural networks (CNNs). Additionally as concerns SVM algortihm the dimensionality reduction step took place using algorithms like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) also the kernel form of PCA - KernelPCA. In the multiclass implementation the use of KPCA following with LDA was essential, in the binary classification we compare the use of PCA and KernalPCA. (e.g., auto-encoders) [4]. 
    The pattern recognition step will be based on Gaussian Mixture Model based classifiers,K-nearest neighbor classifiers, Bayes classifiers, as well as Deep Neural Networks. The Massachussets Eye and Ear Infirmary Dataset (MEEI-Dataset) [5] will be exploited. At the application level, a library for feature extraction and classification in Python will be developed. Credible publicly available resources will be 1used toward achieving our goal, such as KALDI. Comparisons will be made against [6-8].

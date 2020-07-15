# Speech-Signal-Processing-and-Classification

 #### Aristotle University of Thessaloniki - University of Groningen
 
Abstract of my thesis conducted during the 7-8th semester. " Two-class classification problems by analyzing the speech signal. "
 
 
 
   Front-end speech processing aims at extracting proper features from short- term segments of a speech utterance, known as frames. It is a pre-requisite step toward any pattern recognition problem employing speech or audio (e.g., music). Here, we are interesting in voice disorder classification. 
   That is, to develop two-class classifiers, which can discriminate between utterances of a subject suffering from say vocal fold paralysis and utterances of a healthy subject.The mathematical modeling of the speech production system in humans suggests that an all-pole system function is justified [1-3]. As a consequence, linear prediction coefficients (LPCs) constitute a first choice for modeling the magnitute of the short-term spectrum of speech. LPC-derived cepstral coefficients are guaranteed to discriminate between the system (e.g., vocal tract) contribution and that of the excitation. Taking into account the characteristics of the human ear, the mel-frequency cepstral coefficients (MFCCs) emerged as descriptive features of the speech spectral envelope. Similarly to MFCCs, the perceptual linear prediction coefficients (PLPs) could also be derived. The aforementioned sort of speaking traditional features will be tested against agnostic-features extracted by convolutive neural networks (CNNs) (e.g., auto-encoders) [4]. Additionally as concerns SVM algortihm the dimensionality reduction step took place using algorithms like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) also the kernel form of PCA - KernelPCA. In the multiclass implementation the use of (KPCA) following with LDA was essential, in the binary classification we compare the use of PCA and KernelPCA. For experimental purposes Graph Spectral Analysis(IsoMap, LLE) was used for dimensionality reduction followed with Spectral Clustering in order to investigate subsets.
    The pattern recognition step will be based on Gaussian Mixture Model based classifiers,K-nearest neighbor classifiers, Bayes classifiers, as well as Deep Neural Networks. At the application level, a library for feature extraction and classification in Python will be developed. Credible publicly available resources will be used toward achieving our goal, such as KALDI. Comparisons will be made against [5-7].

[1]X. Huang, A. Acero, and H.-W. Hon, Spoken Language Processing. Up-
per Saddle River, N.J.: Pearson Education-Prentice Hall, 2001.

[2] J. R. Deller, J. H. L. Hansen, and J. G. Proakis, Discrete-Time Pro-
cessing of Speech Signals. New York, N.Y.: Wiley-IEEE, 1999.

[3] L. R. Rabiner and R. W. Schafer, Theory and Applications of Digital
Speech Processing. Upper Saddle River, N.J.: Pearson Education- Prentice
Hall, 2011.

[4] Wei-Ning Hsu, Yu Zhang, and James R. Glass, Unsupervised Domain
Adaptation for Robust Speech Recognition via Variational Autoencoder-
Based Data Augmentation,2017, http://arxiv.org/abs/1707.06265.

[5] C. Kotropoulos and G.R. Arce, ”Linear discriminant classifier with re-
ject option for the detection of vocal fold paralysis and vocal fold edema”,
EURASIP Advances in Signal Processing, vol. 2009, article ID 203790, 13
pages, 2009 (DOI:10.1155/2009/203790).

[6] E.Ziogas and C.Kotropoulos, ”Detection of vocal fold paralysis and
edema using linear discriminant classifiers” in Proc. 4th Panhellenic Ar-
tificial Intelligence Conf. (SETN-06), Heraklion, Greece, vol. LNAI 3966,
pp. 454-464, May 19-20, 2006.

[7] M.Marinaki, C.Kotropoulos, I.Pitas, and N.Maglaveras, ”Automatic de-
tection of vocal fold paralysis and edema” in Proc. 8th Int. Conf. Spoken
Language Processing (INTERSPEECH 2004), Jeju, Korea, pp. 537-540, Oc-
tober, 2004.

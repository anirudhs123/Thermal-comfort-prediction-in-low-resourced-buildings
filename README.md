# A-hybrid-deep-transfer-learning-strategy-for-thermal-comfort-prediction-in-buildings
This work entails the usage of a novel hybrid deep transfer learning method for thermal comfort prediction in buildings with no or very little labelled data. We provide a novel Deep Learning architecture that jointly uses CNNs and LSTMs and the concept of Knowledge transfer from source to target domain to accurately predict the thermal comfort in low resourced buildings. We perform extensive ablation study and comparative anlaysis with other state of the art models and out perform them in various quality metrics. For detailed information regarding this work, please visit [our publication](https://www.sciencedirect.com/science/article/abs/pii/S0360132321005345). 

<p align="center">
   <img src="images/Model_arch.jpg" width=500 height=500>
</p>

## Key Contributions
1. A transfer learning based CNN-LSTM (TL CNN-LSTM) model is presented for accurate thermal comfort prediction in buildings with limited modeling data across different climate zones. In the design of TL CNN-LSTM, two significant challenges such as the identification of significant TCPs and imbalanced nature of the data were addressed.
2. The developed model takes input of personal, indoor and outdoor features from the source datasets in specific order and captures the spatio temporal relations for accurate thermal comfort modeling.
3. Extensive experiments on ASHRAE RP-884, Scales project, and Medium US office datasets show that TL CNN-LSTM outperforms the state of-the-art thermal comfort algorithms in terms of various quality metrics (Accuracy, ROC-AUC Score, Mathews Correlation Coefficient)).
4. The studies on the impact of significant TCPs and their different combinations on thermal comfort modeling indicate that TL CNN-LSTM achieves best prediction performance with nine TCPs (PMV, personal, and outdoor environmental factors).
5. The experiments on analyzing the impact of (i) CNN and LSTM layers on TL CNN-LSTM, (ii) CNN-LSTM layers for parameter transfer, and (iii) size of the target dataset on TL CNN-LSTM and CNN-LSTM demonstrates the effectiveness and applicability of the proposed transductive transfer learning based thermal comfort model for buildings with limited modeling data.

Link to paper: https://www.sciencedirect.com/science/article/abs/pii/S0360132321005345

## People

This work has been developed by [Anirudh Sriram](https://github.com/anirudhs123), [Dr. Nivethitha Somu ](https://scholar.google.com/citations?user=q1M0BgIAAAAJ&hl=en), [Prof. Anupama Kowli](https://www.ee.iitb.ac.in/web/people/faculty/home/anu) and [Prof.Krithi Ramamritham ](https://www.iitb.ac.in/en/employee/prof-krithi-ramamritham) from Indian Institute of Technology, Madras and Indian Institute of Technology, Bombay. Ask us your questions at [anirudhsriram30799@gmail.com](mailto:anirudhsriram30799@gmail.com).

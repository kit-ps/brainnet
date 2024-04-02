# BrainNet

This repository contains the code and testbed for the experiments in the publication:

 

 

Fallahi, Matin, Thorsten Strufe, and Patricia Arias-Cabarcos. "BrainNet: Improving Brainwave-based Biometric Recognition with Siamese Networks." 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom).

 

```

@inproceedings{fallahi2023brainnet,

  title={BrainNet: Improving Brainwave-based Biometric Recognition with Siamese Networks},

  author={Fallahi, Matin and Strufe, Thorsten and Arias-Cabarcos, Patricia},

  booktitle={2023 IEEE International Conference on Pervasive Computing and Communications (PerCom)},

  pages={53--60},

  year={2023},

  organization={IEEE}

}

```

# Repository Structure

- **Folders**:
   -  The **"Data"** folder contains preprocessed samples from the datasets described in the paper.
   -  The **"Plots"** folder contains plots generated during the final evaluation in the "Evaluation.py".
   - The **"Similarity Scores"** folder contains similarity scores calculated based on different scenarios, mentioned in the paper, during training and evaluation using the "Dataset_1_2_train" or "Dataset_3_train" code.

- Files:
   - **"Dataset_1_2_train.py"** and **"Dataset_3_train.py"** contain Python files for training networks and storing similarity scores.
   - **"Evaluation.py"** gives you final results based on similarity scores stored previously.
   - **"Network details table.pdf"** represents the parameters of CNN sub-networks in BrainNet and parameters of the encompassing Siamese Network.


   # Datasets Map
- P300:ERP CORE -> D_1
- N400:ERP CORE -> D_2
- P300:bi2015a -> D_3 




# Data

https://i62nextcloud.tm.kit.edu/index.php/s/fj7mrk8SFHi4msQ


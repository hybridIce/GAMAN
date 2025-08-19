# GAMAN
GAMAN: Gated Attention Unit and Mask Attention Network for Traffic Flow Forecasting
This repository contains the implementation of GAMAN, a deep learning framework for short-term traffic flow prediction. GAMAN combines Gated Attention Units (GAU) for temporal modeling and masked attention for spatial dependencies, achieving high performance, computational efficiency, and interpretability on real-world datasets like PeMSD4 and PeMSD8.

Paper Details

Title: Gated attention unit and mask attention network for traffic flow forecasting  
Author: Sen Leng  
Journal: Neural Computing and Applications  
Publisher: Springer Nature  
Received: 26 April 2024  
Accepted: 3 October 2024  
Published: 2025  
Link: #[Full Paper on Springer](https://link.springer.com/article/10.1007/s00521-025-11378-0)

Paper Details

Title: Gated attention unit and mask attention network for traffic flow forecasting
Author: Sen Leng
Journal: Neural Computing and Applications
Publisher: Springer Nature
Received: 26 April 2024
Accepted: 3 October 2024
Published: 2025
DOI: 10.1007/s00521-025-11378-0


@article{leng2025gated,
  title = {Gated attention unit and mask attention network for traffic flow forecasting},
  author = {Leng, Sen},
  journal = {Neural Computing and Applications},
  year = {2025},
  doi = {10.1007/s00521-025-11378-0},
  url = {https://doi.org/10.1007/s00521-025-11378-0},
  publisher = {Springer}
}

## 📦 Installation


**Steps**
1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/hybridIce/GAMAN.git
    cd GAMAN
    ```

2.  (Recommended) Create and activate a new conda virtual environment:
    ```
    conda create -n gaman python=3.8
    conda activate gaman
    ```


3.  Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4.  Prepare datasets:
    Please refer to the LibCity documentation at https://github.com/LibCity/Bigscity-LibCity.
## 🚀 Quick Start


    python run_model.py --task traffic_state_pred --model GAMAN --dataset PeMSD8



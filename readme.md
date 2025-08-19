# GAMAN


## 📦 安装 (Installation)


**步骤 (Steps)**
1.  克隆本仓库到本地：
    ```bash
    git clone https://github.com/hybridIce/GAMAN.git
    cd GAMAN
    ```

2.  （推荐）创建一个新的 conda 虚拟环境并激活它：
    ```
    conda create -n gaman python=3.8
    conda activate gaman
    ```


3.  安装依赖包：
    ```
    pip install -r requirements.txt
    ```
4.  准备数据集
    请参考libcity  https://github.com/LibCity/Bigscity-LibCity
## 🚀 快速开始 (Quick Start)


    ```
    python run_model.py --task traffic_state_pred --model GAMAN --dataset PeMSD8
    ```



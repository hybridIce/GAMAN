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

## 🚀 快速开始 (Quick Start)

这是最重要的部分，用一个最简单的例子告诉用户如何立刻用起来。

python run_model.py --task traffic_state_pred --model GAMAN --dataset PeMSD8

# 3. 训练模型 (这里简化了训练循环)
model.fit(graph_data, features, labels, epochs=100)


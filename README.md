
# GAMAN

A neural network for traffic prediction.  
The code for this paper is implemented based on the [LibCity](https://github.com/LibCity/Bigscity-LibCity) platform.  
You can find the original platform at: https://github.com/LibCity/Bigscity-LibCity

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/hybridIce/GAMAN.git
cd GAMAN
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n GAMAN python==3.9
conda activate GAMAN
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
## Train
```bash
python run_model.py --task traffic_state_pred --model GAMAN --dataset pemsd8
```

## Evalaute
```bash
python run_model.py --task traffic_state_pred --model GAMAN --dataset pemsd8 --exp_id GAMAG_pemsd8
```

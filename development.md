# Instructions

### Create virtual environment 
It is recommended to use Python3.12, since python3.13 is too new for some of the libraries. 
```
python3.12 -m venv .ve
. .ve/bin/activate
```

### Install Requirements.txt 
```
pip install --upgrade pip
pip install -r requirements.txt
```
- Apple Silicon (M1/M2/M3): No extra steps needed. The DeviceManager will automatically use the MPS backend.
- NVIDIA GPU: Ensure you have the appropriate CUDA Toolkit installed (11.8 or 12.1 is recommended).
- CPU Only: The code will default to CPU if no accelerator is found.

### Test & Run
```
python main.py / run main.py file
```

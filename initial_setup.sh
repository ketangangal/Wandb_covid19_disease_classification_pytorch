echo [$(date)]: "Starting InitialSetup...."
export _VERSION_=3.8
echo [$(date)]: "Creating Conda environment with python ${_VERSION_}"
conda create --prefix ./env python=${_VERSION_} -y
echo [$(date)]: "Activating environment..."
source activate ./env
echo [$(date)]: "Creating requirements.txt"
touch requirements.txt
cat > requirements.txt << EOF
jupyter
from_root
pandas
opencv-python
torch
torch vision
pyyaml
matplotlib
EOF
echo [$(date)]: "installing requirements.txt"
pip install -r requirements.txt
#echo [$(date)]: "create directory structure"
#touch config.yaml
#touch setup.py
#mkdir data
#mkdir -p src && touch src/__init__.py
#touch train.py
#touch predict.py

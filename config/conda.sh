conda create --name phon-cxty python=3.6.1
conda activate phon-cxty

conda install -y numpy
conda install -y pandas
conda install -y scikit-learn

conda install -y tqdm
conda install -y matplotlib
pip install seaborn

# This version of NLTK is required for compatibility
pip install https://github.com/nltk/nltk/tarball/model
# conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

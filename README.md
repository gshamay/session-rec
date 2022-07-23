<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="markdown-github.css">
<article class="markdown-body">
<h1>Session-Rec | End Of Session Classification For Session Based Recommendation Systems </h1>
<h2>Author</h2>
Gil Shamay, Ben-Gurion University of the Negev, Israel.
<h2>Introduction</h2>
This Code was forked from and based on the code published at 
<a href=https://github.com/rn5l/session-rec>Github rn5l/session-rec project.</a>
<br>
It is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It
implements a suite of algorithms and baselines for session-based and session-aware recommendation. <br>
This state of the art project, published by Ludewig et al. as part of Empirical analysis of session-based recommendation algorithms.
Their work provided me and other, the ability to use different Session Based Recommendation Algorithms, evaluate and research different aspects and new methods in the field.<br>
<br>
This project extends their code to implement the method we suggested to produce a classification prediction on session based data.<br>
Detailed description of the different parameters, running options and credit for the authors and the developed <br>
can be found in the original code project <a href= https://github.com/rn5l/session-rec/#readme> detailed read me document.</a><br>
<br>
This code was developed as part of a Thesis submitted in partial fulfillment of the requirements for the Master of Sciences degree
at Ben-Gurion University of the Negev, Faculty of Engineering Sciences,  
Department of Software and Information System Engineering, under the supervision of Prof. Lior Rokach.
<br>
<h2>Usage</h2>
We run our experiments on Amazon AES EC2 machines2. We used mainly
c5ad.2xlarge3 machines, with ”Deep Learning Base AMI (Ubuntu 18.04)
Version 48.0” configuration4 and Python 3.7.12. <br>
in this section we will describe how to setup the EC2 machine<br>
<h3>Setup Python</h3>
Setup Python and required libraries
<h3>Download Python 3.7</h3>
sudo apt update<br>
sudo apt install software-properties-common<br>
sudo add-apt-repository ppa:deadsnakes/ppa<br>
sudo apt update<br>
sudo apt install python3.7<br>
python3 --version<br>
<h3>Set installed python3 as default</h3>
sudo update-alternatives --install /usr/bin/python3 python3<br>
/usr/bin/python3.6 1<br>
sudo update-alternatives --install /usr/bin/python3 python3<br>
/usr/bin/python3.7 2<br>
sudo update-alternatives --config python3<br>
python3 -V<br>
<h3>download pip3</h3>
sudo apt install python3-pip<br>
pip3 --version<br>
python3 -m pip install --upgrade pip<br>
upgrade pip<br>
sudo -H pip3 install --upgrade pip<br>
pip3 install --upgrade setuptools pip<br>
<h3>install packages</h3>
pip3 install scipy==1.6.2<br>
pip3 install python-dateutil==2.8.1<br>
pip3 install pytz==2021.1<br>
pip3 install certifi==2020.12.5<br>
pip3 install numpy==1.20.2<br>
pip3 install dill==0.3.3<br>
pip3 install pyyaml==5.4.1<br>
pip3 install networkx==2.5.1<br>
pip3 install scikit-learn==0.24.2<br>
pip3 install numexpr==2.7.3<br>
pip3 install keras==2.3.1<br>
pip3 install six==1.15.0<br>
pip3 install theano==1.0.3<br>
pip3 install pandas==1.2.4<br>
pip3 install psutil==5.8.0<br>
pip3 install pympler==0.9<br>
pip3 install tensorflow==1.14<br>
pip3 install pytables==3.6.1<br>
pip3 install scikit-optimize==0.8.1<br>
pip3 install python-telegram-bot==13.5<br>
pip3 install matplotlib<br>
<h3>Download the Code</h3>
git clone https://github.com/gshamay/session-rec<br>
From now, run all next steps from the code base directory: session-rec.<br>
$cd session-rec<br>
<h3>Download Data</h3>
download ZIP<br>
pip3 install gdown<br>
Download Data<br>
open python3: $python3<br>
>>> import gdown<br>
>>> # DataDownload Yoochoose Data<br>
>>> gdown.download("https://drive.google.com/u/0/uc?id=<br>
19mMXCZeiiK1i4TOLb2VdFPmF0bDqRqcY&export=download","rsc15.zip")<br>
unzip rsc15.zip -d ./data/<br>
>>> # DataDownload Diginetica Data<br>
>>> gdown.download("https://drive.google.com/u/0/uc?id=<br>
1z6Uk9LFYi0i4wEXY2fJ30pX6oF2IZUPs&export=download","digi.zip")<br>
unzip digi.zip -d ./data/<br>
<h3>Running Experiments</h3>
<h4>Using tmux</h4>
It is recommended to use tmux to avoid losing the EC2 session in case of any disconnection<br>
tmux new -s myExperiment<br>
To disconnect from the tmux session press:<br>
Ctrl+b, d<br>
To return to the tmux session run:<br>
tmux attach-session -t myExperiment<br>
To see all running tmux sessions, run:<br>
tmux ls<br>
To terminate all tmux sessions run: pkill -f tmux<br>
<h4>Pre-Process the Data</h4>
There are many yml files in the project, that are ready to be used, for reprocessing the data. <br>
for example, to generate the Yoochoose64 data with 1 aEOS run:<br>
session-rec$ python3 run preprocessing.py<br>
./conf/preprocess/session based/single/rsc15 64 1EOS.yml<br>
<h4>Running the Algorithm</h4>
To train and evaluate the algorithm, run run config.py with a yml that
contain all required configurations. There are many yml files within the
project code. for example, to run sgnn on Yoochoose64 with 1 aEOS run:<br>
session-rec$ python3 run config.py<br>
./conf/in/rsc sgnn 1EOS LR.yml<br>
<h3>8.4 Results</h3>
There are a few result file that are written by the experiment platform:<br>
test single SGNN rsc15 64 1EOS.2-Saver@50.csv - the predictions generated by the SRS model.<br>
plot PrecisionRecall sgnn-best rsc15 64 1EOS.png - the Precision, Recall curve image.<br>
plot PrecisionRecallThresholds sgnn-best rsc15 64 1EOS.png - the Precision,Recall by Thresholds image.<br>
test single SGNN rsc15 64 1EOS.csv - the numerical measurements of the SRS and the classifier.<br>
ThresholdsPrecisionRecall sgnn-best rsc15 64 1EOS.csv - the Precision, Recall and values by Thresholds. clfProbs.csv - the probabilities predictions of the model. br>
clfProbsBaseLine.csv - the probabilities predictions of the baseline. clf.pkl - a pickel of the classifier. clfBaseLine.pkl - a pickel of the baseline.<br>
The results files will appear under the results directory, under a directory with dataset name and then a directory with the algorithm name. for example:<br>
./results/rsc15/rsc15 64 1EOS LR/SGNN<br>
</article>
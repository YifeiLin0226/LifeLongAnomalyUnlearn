# LifeLongAnomalyUnlearn
An unofficial implementation of Lifelong Anomaly Detection Through Unlearning(CCS' 19)[read here](https://dl.acm.org/doi/10.1145/3319535.3363226)

Due to the limited public implementations in the field of lifelong/online log anomaly detection, this repo is created for the study purpose and only the implementation with the HDFS log dataset is made. You are more than welcome to share your insights and point out any code issues.
The paper seems to produce fake/wrong results, especially the threshold part where they claim to set $\tau' = 10^-3, 10^-4, 10^-5$ and predict the sample to be anomaly when $\L(x_t) > \tau'$. $\L(x_t)$ is the cross entropy loss and the result probability for the next gt token has to be greater than 99% if $\L(x_t) = 10^-3$.

## Install Requirements
```
pip install -r requirements.txt
```
or
```
pip install torch logparser3
```

## Download Datasets
1. Download and extract HDFS dataset from [here](https://zenodo.org/records/8196385)
2. In main.py, set the **input_dir** to the directory of **HDFS.log**, set the **out_dir** to the directory of storing the parsed and processed data.
   ```
   input_dir =   
   output_dir =
   ```

## Run the program
```
python -m main
```
Feel free to adjust the params in **options** variable

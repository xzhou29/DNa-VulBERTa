
mkdir /scratch/dna_data_pretraining_2/devign
mkdir /scratch/dna_data_pretraining_2/mvdsc
mkdir /scratch/dna_data_pretraining_2/d2a
mkdir /scratch/dna_data_pretraining_2/poj
mkdir /scratch/dna_data_pretraining_2/sysevr

python src/data/DP_devign.py --input /scratch/dna_data_raw/devign/dataset.json --output /scratch/dna_data_pretraining_2/devign/all.pkl --workers 16 --all 0 


python src/data/DP_mvdsc.py --input /scratch/dna_data_raw/mvdsc/raw_train.csv.gz --output /scratch/dna_data_pretraining_2/mvdsc/train.pkl --workers 16 


python src/data/DP_d2a.py --input /scratch/dna_data_raw/d2a/DAX_D2ALBData/function/d2a_lbv1_function_train.csv --output /scratch/dna_data_pretraining_2/d2a/d2a_train.pkl --workers 16 


python src/data/DP_poj.py --input /scratch/dna_data_raw/Clone-detection-POJ-104/dataset/train.jsonl --output /scratch/dna_data_pretraining_2/poj/poj_train.pkl --workers 16 


python src/data/DP_sysevr.py --input /scratch/dna_data_raw/sysevr/API_function_call/API_function_call.txt --output /scratch/dna_data_pretraining_2/sysevr/api_train.pkl --workers 16 --all 0 


python src/data/DP_sysevr.py --input /scratch/dna_data_raw/sysevr/Arithmetic_expression/Arithmetic_expression.txt --output /scratch/dna_data_pretraining_2/sysevr/ae_train.pkl --workers 16 --all 0 


python src/data/DP_sysevr.py --input /scratch/dna_data_raw/sysevr/Array_usage/Array_usage.txt --output /scratch/dna_data_pretraining_2/sysevr/au_train.pkl --workers 16 --all 0 


python src/data/DP_sysevr.py --input /scratch/dna_data_raw/sysevr/Pointer_usage/Pointer_usage.txt --output /scratch/dna_data_pretraining_2/sysevr/pu_train.pkl --workers 16 --all 0
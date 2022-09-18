


python src/data/DP_devign.py --input ../vul_dataset/dna_raw_data/devign/dataset.json --output ../vul_dataset/dna_data/devign_train.pkl --workers 28 --iter 5

python src/data/DP_poj.py --input ../vul_dataset/dna_raw_data/Clone-detection-POJ-104/dataset/train.jsonl --output ../vul_dataset/dna_data/poj_train.pkl --workers 28 --iter 5

python src/data/DP_mvdsc.py --input ../vul_dataset/dna_raw_data/mvdsc/raw_train_combined.csv.gz --output ../vul_dataset/dna_data/mvdsc_train.pkl --workers 28 --iter 5

python src/data/DP_d2a.py --input ../vul_dataset/dna_raw_data/d2a/DAX_D2ALBData/function/d2a_lbv1_function_train.csv --output ../vul_dataset/dna_data/d2a_train.pkl --workers 28 --iter 5



python src/data/DP_sysevr.py --input ../vul_dataset/dna_raw_data/sysevr/API_function_call/API_function_call.txt --output ../vul_dataset/dna_data/sysevr_apifc.pkl --workers 28 --iter 5

python src/data/DP_sysevr.py --input ../vul_dataset/dna_raw_data/sysevr/Arithmetic_expression/Arithmetic_expression.txt --output ../vul_dataset/dna_data/sysevr_ae.pkl --workers 28 --iter 5

python src/data/DP_sysevr.py --input ../vul_dataset/dna_raw_data/sysevr/Array_usage/Array_usage.txt --output ../vul_dataset/dna_data/sysevr_au.pkl --workers 28 --iter 5

python src/data/DP_sysevr.py --input ../vul_dataset/dna_raw_data/sysevr/Pointer_usage/Pointer_usage.txt --output ../vul_dataset/dna_data/sysevr_pu.pkl --workers 28 --iter 5



python src/data/DP_draper.py --input ../vul_dataset/dna_raw_data/draper/vdisc_train.hdf5 --output ../vul_dataset/dna_data/draper_train.pkl --workers 28 --iter 5


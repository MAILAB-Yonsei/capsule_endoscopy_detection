#!/usr/bin/env bash

# 각 메서드 별로 흩어져 있는 11개의 csv 파일을 하나의 디렉토리로 취합합니다.
# 취합하는 디렉토리는 /csv_results 입니다.
# 이미 학습한 모델로 부터 inference한 각 메서드의 결과 csv 파일들은 /csv_results_pretrained 에 위치하여 있습니다.
python python_codes/gathering_csv.python

# 모든 STEP1~STEP4 를 거쳐서 출력된 11개의 csv 파일에 대해 WBF 앙상블을 수행하여 최종 결과를 저장합니다.
# 저장 경로는 /final.csv 입니다.
python python_codes/ensemble_test.py

# pretrained 된 모델에서 이미 출력해 놓은 11개의 csv 파일에 대해 WBF 앙상블을 수행하여 최종 결과를 저장합니다.
# 저장 경로는 /final_pretrained.csv 입니다.
# 위의 final.csv와 final_pretrained.csv가 거의 동일한 것을 확인합니다.
python python_codes/ensemble_test_pretrained.py
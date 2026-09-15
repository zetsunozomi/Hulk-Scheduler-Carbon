log_path_1="/lus/eagle/projects/Local-LLM/shuyuanfan/carbon/src/model/policy-gradient-moe/exp2/log_gpt345M_1.0.txt"
log_path_2="/lus/eagle/projects/Local-LLM/shuyuanfan/carbon/src/model/policy-gradient-moe/exp1/log_gpt345M_1.0.txt"
echo "---------------------Analyzing log ${log_path_1}---------------------"
python analyze_validation_log.py $log_path_1 > analyze_log_1.txt
echo "---------------------Analyzing log ${log_path_2}---------------------"
python analyze_validation_log.py $log_path_2 > analyze_log_2.txt
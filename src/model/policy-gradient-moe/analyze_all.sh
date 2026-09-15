exp_idx=12
data_dir="/pscratch/sd/s/syfan/carbon/src/model/policy-gradient-moe/exp${exp_idx}"
# 0.0 0.1 0.2 0.4 0.6 0.9 

for carbon_weight in 0.99; do
    echo "---------------------Analyzing carbon weight ${carbon_weight}---------------------"
    python analyze_validation_log.py $data_dir/validation_gpt345M_${carbon_weight}.txt
done > analyze_${exp_idx}.txt
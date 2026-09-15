PYTHON_ENV="/pscratch/sd/s/syfan/conda/envs/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

LOG_PATH="/pscratch/sd/s/syfan/carbon/src/model/policy-gradient-moe/exp12/log_gpt345M_0.99.txt"
OUT_DIR="/pscratch/sd/s/syfan/carbon/src/model/policy-gradient-moe/exp12"
mkdir -p $OUT_DIR

START_EPISODE=800
END_EPISODE=900

# 0.0 0.1 0.2 0.4 0.6 0.9 
for carbon_weight in 0.9; do
    echo "Analyzing submit times for carbon weight ${carbon_weight}"
    CMD="$PYTHON_ENV start_time_stats.py $LOG_PATH $OUT_DIR/start_time_stats_${carbon_weight}.png --start_episode $START_EPISODE"
    
    if [ ! -z "$END_EPISODE" ]; then
        CMD="$CMD --end_episode $END_EPISODE"
    fi
    
    echo "Running: $CMD"
    $CMD
done
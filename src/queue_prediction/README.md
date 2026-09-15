# Queue Prediction Model Training
## data generation
off_data_gen.py will generate offline data for queue prediction: given a machine state and node number, predict the queue wait time of a job.
It will generate a json file for each node number, and each json file contains a list of snapshots of machine states.
* num_samples: the number of snapshots to generate
* interval: the interval between snapshots
* od & cd: the output directory and config directory
* workload: the workload file
* start_time: the start time of the workload
* warmup_len: the warmup length (The simulator will run 48 hours as hardcoded.)
* workload_len: the workload length
* node: the node number
## read the data
You may verify the data by running read_json.py.
--- Data Summary ---
Total Samples: 925

--- Data Structure ---
Type: list, Length: 925
  Sample item 0:
    Type: list, Length: 289
      Sample item 0:
        Type: dict, Keys: ['sim_time', 'pending_list', 'running_list']
          Key: 'sim_time'
            Type: str, Value (preview): 2024-08-01T20:00:00
          Key: 'pending_list'
            Type: list, Length: 0
          Key: 'running_list'
            Type: list, Length: 0
      ... (middle items omitted) ...
      Sample item 288 (Last Item):
        Type: dict, Keys: ['job_id', 'queue_wait_sec', 'nodes']
          Key: 'job_id'
            Type: str, Value (preview): 1000000
          Key: 'queue_wait_sec'
            Type: int, Value (preview): 3900
          Key: 'nodes'
            Type: int, Value (preview): 4

In the dataset: 925: number of samples
In a sample:289: 48 hours, 1 snapshots per 10 minutes, 288 snapshots per hour + 1 final test job
In a snapshot: 3 keys: sim_time, pending_list, running_list
In a test job: 3 keys: job_id, queue_wait_sec, nodes
## Prediction Model Validation
Not tested yet, but the online_validate.py script should be used to validate a predition model.

# Data
The log files here contains the slurm job logs. **Currently we use new/filtered_iw_log.log and new/filtered_iw_validate.log for agent training and validation, following the practice of the previous Ciri Paper** 
## Cluster Setting
There are 84 compute nodes in Frontera RTX, each has 4 RTX 5000 GPUs.  
There are 88 compute nodes in Longhorn V100, each has 4 V100 GPUs.  
There are 72 compute nodes in Longhorn gpu-a100, each has 3 A100 GPUs.  
## Log Format
The log is the format of:  
JobID,JobName,UID,start,submit,end,Priority,TimelimitRaw,ExitCode,NNodes
## Time span and number of jobs
The data range of Frontera RTX is from 2019-12-04 to 2021-08-20.  
The data range of Longhorn V100 is from 2019-11-04 to 2021-08-20.  
The data range of Lonestar6 is from 2022-08-01 to 2023-01031.  

filtered/filtered-frontera-rtx.log: 175090  
filtered/filtered-longhorn-v100.log: 65017  
filtered/filtered-ls6-new.log: 24777  
filtered/filtered-ls6.log: 17568  
new/filtered_iw_log.log: 69464  
new/filtered_iw_validate.log: 18908  
raw/frontera-rtx.log: 175229  
raw/longhorn-v100.log: 76046  
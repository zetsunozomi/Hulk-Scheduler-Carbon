import random
import logging
from datetime import datetime, timedelta
from collections import deque

from sim.simulator import Simulator, Mode
from sim.simparser import SimParser
from sim.job import Job
from carbon_intensity_lookup import get_unit_carbon_intensity

logging.basicConfig(level=logging.INFO)

class Env:
    
    def __init__(self,
                 job_log,             
                 slurm_config,       
                 backfill_config,   
                 sim_start_time,      
                 log_start_time,     
                 log_end_time,        
                 warmup_len,       
                 sim_len,            
                 sim_window,
                 seed,
                 job_node=4):
      
        self.slurm_config = SimParser.load_slurm_config(slurm_config)
        self.backfill_config = SimParser.load_backfill_config(backfill_config)
        
        
        self.sim = Simulator(mode=Mode.MINUTES)
        self.sim.load_jobs(job_log)  
        
        
        self.sim_start_time = datetime.strptime(sim_start_time, "%Y-%m-%dT%H:%M:%S")
        self.log_start_time = datetime.strptime(log_start_time, "%Y-%m-%dT%H:%M:%S")
        self.log_end_time   = datetime.strptime(log_end_time,   "%Y-%m-%dT%H:%M:%S")

        self.warmup_len = warmup_len
        self.sim_len    = sim_len  
        self.sim_window = sim_window
        self.random_seed = seed
        self.job_node = job_node
        
        
        self.history_buffer = deque(maxlen=288)
        self._random_set = False

    def reset(self):
        logging.info("Resetting")
        start_time = self.log_start_time
        end_time = self.log_end_time

        if not self._random_set:
            random.seed(self.random_seed)
            self._random_set = True
        time_offset = random.random()
        logging.info(f"Time offset: {time_offset}")
        
        # Calculate the actual target start time for the episode
        target_sim_start_time = timedelta(minutes=int(time_offset * ((end_time - start_time).total_seconds()) / 60))
        target_sim_start_time = target_sim_start_time + start_time
        
        # We need to start 'warmup_len' days earlier
        warmup_duration_days = self.warmup_len
        warmup_start_time = target_sim_start_time - timedelta(days=warmup_duration_days)
        
        logging.info(f"Target Start Time: {target_sim_start_time}")
        logging.info(f"Warmup Start Time: {warmup_start_time}")
        
        sim_end_time = target_sim_start_time + timedelta(days=self.sim_len)

        self.sim.reset()
        # Initialize scheduler slightly before or at warmup_start_time
        # Match data in /pscratch/sd/s/syfan/carbon/src/queue_prediction/offline_data_gen.py
        self.sim.init_scheduler(88, self.slurm_config, warmup_start_time, self.backfill_config)
        
        # Load jobs from warmup_start_time to sim_end_time
        jobs_log = self.sim.find_jobs(warmup_start_time, sim_end_time)
        self.sim.submit_job_internal(jobs_log)
        
        # Run the simulator through the warmup period, filling the history buffer
        logging.info(f"Running warmup for {warmup_duration_days} days to populate history...")
        
        self.history_buffer.clear()
        
        # We need to fill the history buffer with snapshots during warmup.
        # The history buffer expects snapshots at a certain cadence. 
        # Typically the simulation runs in minutes.
        # We will step through the warmup period in increments (e.g., 10 minutes) 
        # and capture snapshots, similar to how step_time does it.
        
        warmup_end_time = warmup_start_time + timedelta(days=warmup_duration_days)
        step_delta = timedelta(minutes=10) # 10 minute steps for history resolution
        
        while self.sim._time < warmup_end_time:
             # Capture snapshot
             current_snapshot = {
                "pending_list": self._get_pending_jobs(),
                "running_list": self._get_running_jobs()
             }
             self.history_buffer.append(current_snapshot)

             # Run simulator for step_delta
             self.sim.run_time(step_delta)
        
        # Verify we reached the target time
        current_sim_time = self.sim._time
        
        self.sim_start_time = current_sim_time 
        logging.info(f"Warmup complete. Current Sim Time: {self.sim_start_time}")
        logging.info(f"History buffer size: {len(self.history_buffer)}")
      
        # Ensure buffer is full (pad with latest if needed, though 2 days @ 10mins > 288)
        self._fill_history_buffer()

    def _fill_history_buffer(self):
        while len(self.history_buffer) < 288:
            self.history_buffer.append({
                "pending_list": [],
                "running_list": []
            })

    def step_time(self, dt: timedelta):
        self.sim.run_time(dt)
        
        current_snapshot = {
            "pending_list": self._get_pending_jobs(),
            "running_list": self._get_running_jobs()
        }
        self.history_buffer.append(current_snapshot)

    def _get_pending_jobs(self):
        pending_list = []
        for job_pend in self.sim._scheduler.pending_queue:
            wait_sec = 0.0
            if self.sim._time and job_pend.submit:
                wait_sec = (self.sim._time - job_pend.submit).total_seconds()

            if isinstance(job_pend.time_limit, timedelta):
                tlimit_in_minutes = job_pend.time_limit.total_seconds() / 60.0
            else:
                tlimit_in_minutes = job_pend.time_limit

            pending_list.append({
                "nodes": job_pend.nodes,
                "time_limit": tlimit_in_minutes,
                #"queue_wait_sec": wait_sec,
                "run_sec": 0,
                "wait_sec": wait_sec
            })
        return pending_list

    def _get_running_jobs(self):
        running_list = []
        run_dict = self.sim._scheduler.running_jobs_state
        for j_id, j_log in run_dict.items():
            run_sec = j_log.finish_time.total_seconds() if j_log.finish_time else 0.0
            wait_sec = 0.0
            if j_log.start and j_log.job.submit:
                wait_sec = (j_log.start - j_log.job.submit).total_seconds()

            if isinstance(j_log.job.time_limit, timedelta):
                tlimit_in_minutes = j_log.job.time_limit.total_seconds() / 60.0
            else:
                tlimit_in_minutes = j_log.job.time_limit

            running_list.append({
                "nodes": j_log.job.nodes,
                "time_limit": tlimit_in_minutes,
                "queue_wait_sec": wait_sec,
                "run_sec": run_sec,
                "wait_sec": wait_sec
            })
        return running_list

    def get_current_data_input(self):
        while len(self.history_buffer) < 288:
            self.history_buffer.appendleft({
                "pending_list": [],
                "running_list": []
            })
        
        curr_time = self.get_current_time()
        # Fallback if sim not started or time is None, though usually it is set.
        if curr_time is None:
            curr_time = self.sim_start_time
            
        # Calculate time of day (e.g. 1:30 -> 1.5)
        # curr_time is a datetime object
        time_of_day = curr_time.hour + curr_time.minute / 60.0
        
        # Get Current Carbon Intensity
        # We need the current hour index.
        current_intensity = get_unit_carbon_intensity(curr_time, curr_time.hour)
            
        return {
            'time_series': list(self.history_buffer),
            'job_info': {
                "time_limit": 2880,  # 48小时 => 2880分钟
                "nodes": self.job_node
            },
            'time_of_day': time_of_day,
            'carbon_intensity': current_intensity
        }

    def submit_job_external(self, job_lst):
        self.sim.submit_job_external(job_lst)

    def run_end(self, job_id):
        return self.sim.run_end(job_id)

    def get_current_time(self):
        return self.sim._time

    def get_start_time(self):
        return self.sim_start_time

    def get_simulator_length_days(self):
        return self.sim_len

    def get_scheduler(self):
        return self.sim._scheduler
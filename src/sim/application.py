import logging
import math
import sys
from datetime import timedelta
from sim.job import Job

logging.basicConfig(level=logging.INFO)


NODE_HOUR_FACTOR_345M = {
    4: 1.0,
    8: 0.6754,
    16:0.3868,
    32:0.2184
}
NODE_HOUR_FACTOR_1p5B = {
    4: 1.0,
    8: 0.7105,
    16:0.4925,
    32:0.2995
}
NODE_HOUR_FACTOR_GPT_LARGE = {
    4: 1.0,
    8: 0.664,
    16:0.401,
    32:0.216
}
#GPT-345M runs on 16 nodes for 22.6hrs
#GPT-large runs on 16 nodes for 42.5 hrs
#GPT-1.5B runs on 16 nodes for 110.0 hrs. 

#GPT-2.0B runs on 16 nodes for 156.0 hrs.
class Application:
    def __init__(self, app_type="gpt-345m", base_run_h=6):
        self.app_type = app_type
        self.base_run_h = base_run_h  
        self.factor_map = {}
        
        if app_type == "gpt-345m":
            self.factor_map = NODE_HOUR_FACTOR_345M
            hrs_at_16 = 22.6
        elif app_type == "gpt-1.5b":
            self.factor_map = NODE_HOUR_FACTOR_1p5B
            hrs_at_16 = 110.0
        elif app_type == "gpt-large":
            self.factor_map = NODE_HOUR_FACTOR_GPT_LARGE
            hrs_at_16 = 42.5
        else:
            print(f"Unknown app_type {app_type}")
            quit()

        self._node = 32
        # Total node hours calculated as the consumption at 16 nodes converted to 4-node equivalent
        # Using the formula implied by user: nodes * time * factor
        # We use the provided benchmarks at 16 nodes.
        self.total_node_hours = 16 * hrs_at_16 * self.factor_map[16]
        self.consumed_node_hours = 0.0

    def reset(self):
        self.consumed_node_hours = 0.0

    def can_submit_next_job(self, env, node_count):
        remaining = self.total_node_hours - self.consumed_node_hours
        logging.info(f"check submission: node {node_count},remaining {remaining:.2f}")
        if remaining > 0:
            return True, remaining
        else:
            return False, remaining

    def create_new_job(self, action_idx, env, next_job_id=9999, base_run_h=None):
        if base_run_h is None:
            base_run_h = self.base_run_h
            
        node_map = [4, 8, 16, 32]
        chosen_node = node_map[action_idx]
        limit_minutes = base_run_h*60
        job_id_str = f"auto_{next_job_id}"
        
        new_job = Job(
            job_id=job_id_str,
            nodes=chosen_node,
            start=None,
            submit=env.get_current_time(), 
            end=None,
            limit=limit_minutes,
            priority=1000,
            priority_valid=False
        )
        return new_job
    def get_nodehour_left(self):
        return self.total_node_hours - self.consumed_node_hours
    def run_job_and_measure_reward(self, env, new_job):
        chosen_node = new_job.nodes
        logging.info(f"submit test job, chosen node is:{chosen_node}")
        
        hour_factor = self.factor_map.get(chosen_node, 1.0)  
        test_job_consumption = chosen_node * self.base_run_h * hour_factor 
        logging.info(f"Job {new_job.job_id}: distribute {chosen_node} nodes,")
        
        if self.total_node_hours - self.consumed_node_hours > test_job_consumption:
            effective_run_time = self.base_run_h
            self.consumed_node_hours += test_job_consumption
            env.submit_job_external([new_job])
            #print(f"env time before run end: {env.get_current_time()}")
            env.run_end(new_job.job_id)
            #print(f"env time after run end: {env.get_current_time()}")
            if not new_job.log or (not new_job.log.start) or (not new_job.log.end):
                logging.warning(f"[Application] job {new_job.job_id} missing log or start/end => fallback=0 => done.")
                return 0.0, True
            
            start_t = new_job.log.start
            end_t   = new_job.log.end
            submit_t= new_job.submit
            
            queue_wait_sec = (start_t - submit_t).total_seconds()
            run_sec        = (end_t   - start_t).total_seconds()

            print(f"submit{submit_t}")
            print(f"start{start_t}")
            print(f"end{end_t}")
            sys.stdout.flush()
            queue_wait_h = queue_wait_sec / 3600.0
            run_time_h   = run_sec / 3600.0
            time_delta_until_endtime = end_t - env.get_current_time()
            # An error of 1 minute happens here. We could ignore it when number of total jobs is not big.
            # Also, the simulator runs at minimum scale, which makes the error ineviable.
            #minutes = int((time_delta_until_endtime).total_seconds() // 60)
            #print(f"Here, we actually need to proceed {minutes} minutes")
            env.step_time(time_delta_until_endtime)
            new_sim_time = env.get_current_time()

            reward = - (queue_wait_h + run_time_h)
        else:
            # after this else happens, the scheduler will be reset.
            # thus no need to change.
            effective_run_time = (self.total_node_hours - self.consumed_node_hours) / (chosen_node * self.factor_map.get(chosen_node, 1.0))
            # new_job.Job.end   = new_job.Job.start + exec_time
            env.submit_job_external([new_job])
            env.run_end(new_job.job_id)
            
            start_t = new_job.log.start
            end_t   = new_job.log.end
            submit_t= new_job.submit

            queue_wait_sec = (start_t - submit_t).total_seconds()

            queue_wait_h = queue_wait_sec / 3600.0

            print(f"submit{submit_t}")
            print(f"start{start_t}")
            print(f"end{end_t}")
            
            reward = - (queue_wait_h + effective_run_time)
            self.consumed_node_hours += test_job_consumption
            
        done = ((self.total_node_hours - self.consumed_node_hours)<= 0)
        return reward, effective_run_time, done

        
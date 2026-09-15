import argparse
import datetime
import json
import logging
import os
import sys
import pickle

import numpy

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)
from sim import *
import sim
import ray
import math


def pre_simulate(simulator_start_time, workload_start_time, workload_end_time, slurm_config, backfill_config,
                 workload_path,queue_time_len, node, output_json="simulate.json"):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    filter_kernel=None
    pivot=None
    jobs = SimParser.parse_job(workload_path, pivot, filter_kernel)

    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs

    slurm_simulator.init_scheduler(88, slurm_config, simulator_start_time, backfill_config, queue_time_len)

    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)

    total_hours = 48
    sim_end_time = simulator_start_time + timedelta(hours=total_hours)
    step_minutes = 10

    occupant_data= [] 

    while slurm_simulator.sim_time < sim_end_time:
        sim_state = slurm_simulator.output_state()
        mini_batch_q = []
        mini_batch_m = []

        # pending
        for j_pend in sim_state.pending_jobs:
            wait_sec = int((slurm_simulator.sim_time - j_pend.submit).total_seconds())
            mini_batch_q.append({
                "job_id": j_pend.job_id,
                "time_limit": j_pend.time_limit,
                "nodes": j_pend.nodes,
                "wait_sec": wait_sec
            })

        # running
        for _, job_log in sim_state.running_jobs.items():
            wait_sec = int((job_log.start - job_log.job.submit).total_seconds())
            run_sec  = int(job_log.finish_time.total_seconds())
            mini_batch_m.append({
                "job_id": job_log.job.job_id,
                "time_limit": job_log.job.time_limit,
                "nodes": job_log.job.nodes,
                "queue_wait_sec": wait_sec,
                "run_sec": run_sec
            })

        occupant_data.append({
            "sim_time": slurm_simulator.sim_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pending_list": mini_batch_q,
            "running_list": mini_batch_m
        })

        # 前进10min
        slurm_simulator.run_time(timedelta(minutes=step_minutes))

    print(f"[pre_simulate] => collected {len(occupant_data)} snapshots of occupant_data.")
    return occupant_data, slurm_simulator

def simulate_per_sample(sim_instance: Simulator, test_job_start: datetime, node: int,
                       sample_time_step=144, output_json="machine_states.json"):
    # 1) 构造测试作业
    test_job_id = "1000000"
    job_1 = Job(
        test_job_id,
        node,
        start=None,
        submit=test_job_start,
        end=None,
        limit=2880,  
        priority=6000,
        priority_valid=False
    )

    sim_instance.submit_job_external([job_1])

   
    step_minutes = 10

    queue_wait_sec = None

    max_hours = 72
    
    end_time = sim_instance.sim_time + timedelta(hours=max_hours)

    while True: #sim_instance.sim_time < end_time:
        sim_instance.run_time(timedelta(minutes=step_minutes))
        test_job_log = None
        for j_log in sim_instance._scheduler.job_logs:
            if j_log.job.job_id == test_job_id:
                test_job_log = j_log
                break

        if test_job_log is not None and test_job_log.start is not None:
            print(f"test_job_log.start: {test_job_log.start}  test_job_log.job.submit: {test_job_log.job.submit}")
            wait_sec = (test_job_log.start - test_job_log.job.submit).total_seconds()
            queue_wait_sec = int(wait_sec)
            print(f"[simulate_per_sample] => test job started! queue_wait={queue_wait_sec}, node={node}")
            break
    test_job_data = {
        "job_id": test_job_id,
        "queue_wait_sec": queue_wait_sec,
        "nodes": node
    }
    print(test_job_data)
    return test_job_data
    
@ray.remote
def simulate_per_thread(init_time, i, slurm_config, backfill_config, start_time_interval, workload_path,
                        workload_len, warmup_len,node=1, output_dir='./'):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.ERROR)
    data = []
    simulator_start_time = init_time + timedelta(hours=i * start_time_interval)
    workload_start_time = simulator_start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)

    # B) pre_simulate
    occupant_data, sim_instance= pre_simulate(
        simulator_start_time,
        workload_start_time,
        workload_end_time,
        slurm_config,
        backfill_config,
        workload_path,
        queue_time_len= 100,
        node= node
    )

    # C) 预热完成 => simulator此时 sim_time= simulator_start_time+48h
    end_warmup= sim_instance.sim_time

    # D) 立即提交测试job
    test_data = simulate_per_sample(sim_instance, end_warmup, node=node)

    # E) 合并 occupant_data + test_data => 写 occupant_json
    occupant_data.append(test_data)
    print(f"[simulate_per_thread] => total {len(occupant_data)} snapshots.")

    return occupant_data


def simulate_single(init_time, i, slurm_config, backfill_config, num_probe, start_time_interval, workload_path,
                    workload_len, warmup_len, early_age, baseline_avg_queue_time_len, sample_time_window, pivot,
                    mode="default", regr_model=None, quantile_model=None, node=1):
    data = []
    simulator_start_time = init_time + timedelta(hours=i * start_time_interval)
    workload_start_time = simulator_start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)
    lower_bound, upper_bound, avg_time, waiting_queue_size = pre_simulate(simulator_start_time, workload_start_time,
                                                                          workload_end_time,
                                                                          slurm_config, backfill_config, workload_path,
                                                                          warmup_len,
                                                                          regr_model, baseline_avg_queue_time_len,
                                                                          pivot, node)
    if mode == "baseline_avg":
        sample_point = [upper_bound - timedelta(minutes=avg_time)]
    elif mode == "baseline_reactive":
        sample_point = [upper_bound]
    elif mode == "baseline_quantile_regr":
        if quantile_model is None:
            sample_point = [upper_bound]
        else:
            test = [(avg_time / 60) * waiting_queue_size]
            test = numpy.array(test).reshape(-1, 1)
            pred_value = int(quantile_model.predict(test)[0] * 60)
            sample_point = [upper_bound - timedelta(minutes=pred_value)]
    else:
        interval = (upper_bound - lower_bound).total_seconds() / (num_probe - 1.0)
        sample_point = [(lower_bound + timedelta(seconds=interval * j)).replace(second=0) for j in range(0, num_probe)]
    for point in sample_point:
        data.append(
            simulate_per_sample(point, simulator_start_time, workload_start_time, workload_end_time,
                                slurm_config, backfill_config, workload_path, warmup_len, regr_model, early_age,
                                sample_time_window, pivot, node)[0][0])
    return data, avg_time


def simulate(num_samples, start_time_interval, parallel, workload_name, sim_init_time, workload_len,
             warmup_len, cpu_cores, output_dir,slurm_config,
             backfill_config, node=1):
    if not parallel:
        # 顺序
        for i in range(num_samples):
            print(f"[simulate_main] => job {i+1}/{num_samples}")
            simulator_start_time= sim_init_time + timedelta(hours= i* start_time_interval)
            occupant_json= os.path.join(output_dir, f"occupant_48h_{i}.json")

            # 1) pre_simulate
            sim_instance= pre_simulate(
                simulator_start_time,
                simulator_start_time,
                simulator_start_time+ timedelta(days=workload_len),
                slurm_config,
                backfill_config,
                workload_path= os.path.join(PROJECT_ROOT, "workload", workload_name),
                queue_time_len= 100,
                node=node,
                output_json= occupant_json
            )
            # 2) run_test_job
            run_test_job(sim_instance, sim_instance.sim_time, node, output_json= occupant_json)
    else:
        # parallel => Ray
        file = os.listdir("../workload")
        file = list(filter(lambda x: x != workload_name, file))
        file = list(map(lambda x: "/workload/" + x, file))

        if cpu_cores == -1:
            ray.init(runtime_env={"py_modules": [sim], "working_dir": "../",
                                  "excludes": ["/test/", "/data/", "/misc/", "/queue_prediction/json_data/"] + file})
        else:
            ray.init(runtime_env={"py_modules": [sim], "working_dir": "../",
                                  "excludes": ["/test/", "/data/", "/misc/", "/queue_prediction/json_data/"] + file},
                     num_cpus=cpu_cores)

        tasks=[]
        for i in range(num_samples):
            tasks.append(
                simulate_per_thread.remote(
                    sim_init_time,
                    i,
                    slurm_config,
                    backfill_config,
                    start_time_interval,
                os.path.join(PROJECT_ROOT, "workload", workload_name),
                    workload_len,
                    warmup_len,
                    node
                )
            )
        finish_count=0
        total_finish_time= datetime.today() - datetime.today()
        results= []

        while len(tasks):
            wait_time= datetime.today()
            done_id, tasks= ray.wait(tasks)
            occupant_data_i= ray.get(done_id[0])
            finish_count+=1
            total_finish_time+= (datetime.today()- wait_time)

            results.append(occupant_data_i)

            avg_complete_sec= total_finish_time.total_seconds()/ finish_count
            expected_remaining= avg_complete_sec*(num_samples- finish_count)
            print(f"[simulate] => done run {finish_count}/{num_samples}, "
                  f"avg= {int(avg_complete_sec//3600)}:"
                  f"{int((avg_complete_sec%3600)//60)}:"
                  f"{int(avg_complete_sec%60)}, remain= "
                  f"{int(expected_remaining//3600)}:"
                  f"{int((expected_remaining%3600)//60)}:"
                  f"{int(expected_remaining%60)}")

        ray.shutdown()

        # results => [ occupant_data_0, occupant_data_1, ... occupant_data_(N-1) ]
        all_data= results
        outpath= os.path.join(output_dir, "all_data.json")
        with open(outpath, "w") as fw:
            json.dump(all_data, fw, indent=2)
        print(f"[simulate] => wrote {len(all_data)} runs to {outpath}")


def main():
    prev_time = datetime.today()
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("-parallel", action="store_true", default=False)
    parser_arg.add_argument("-num_samples", type=int, default=2)
    parser_arg.add_argument("-interval", "--start_time_interval", type=int, default=6)
    parser_arg.add_argument("-workload", "--workload_name", type=str, default="filtered-longhorn-v100.log")
    parser_arg.add_argument("-start_time", type=lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"),
                            default="2021-03-01T00:00:00")
    parser_arg.add_argument("-workload_len", type=int, default=5)
    parser_arg.add_argument("-od", "--output_dir", default="./top")
    parser_arg.add_argument("-cd", "--config_dir", default="../test/test_data")
    parser_arg.add_argument("-warmup_len", type=int, default=2)
    parser_arg.add_argument("-file_split", action="store_true", default=False)
    parser_arg.add_argument("-file_num", type=int, default=4)
    parser_arg.add_argument("-philly_tl", "--philly_time_limit", default=None)
    parser_arg.add_argument("-philly", action="store_true", default=False)
    parser_arg.add_argument("-ref_trace", default="../workload/filtered-frontera-rtx.log")
    parser_arg.add_argument("-cpu_cores", type=int, default=-1)
    parser_arg.add_argument("-early_age", action="store_true", default=False)
    parser_arg.add_argument("-ray_reset", action="store_true", default=False)
    parser_arg.add_argument("-quantile_model", default=None)
    parser_arg.add_argument("-baseline_avg_num", type=int, default=100)
    parser_arg.add_argument("-window", type=int, default=144)
    parser_arg.add_argument("-job_duration_pivot", type=float, default=None)
    parser_arg.add_argument("-start_index", type=int, default=0)
    parser_arg.add_argument("-job_filter", type=str, default=None)
    parser_arg.add_argument("-node", type=int, default=1)
    args = parser_arg.parse_args()

    if args.job_filter is not None:
        with open(args.job_filter) as f:
            job_filter = json.load(f)["filter_range"]
        job_filter = list(map(lambda x: list(map(lambda y: datetime.strptime(y, "%Y-%m-%dT%H:%M:%S"), x)), job_filter))
    else:
        job_filter = None

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.ERROR)
    print("--------------------------------------------------------------------------------------------------------")
    print("Simulation starts...")
    os.makedirs(args.output_dir, exist_ok=True)
    print("-----------------------")
    regr = None
    

    slurm_config = simparser.SimParser.load_slurm_config(os.path.join(args.config_dir, "slurm_config.json"))
    backfill_config = simparser.SimParser.load_backfill_config(os.path.join(args.config_dir, "backfill_config.json"))
    if args.ray_reset and args.file_split:
        file_step = math.ceil(args.num_samples / args.file_num)
        for i in range(0, args.file_num):
            print("-----------------------")
            print(f"Partially Simulate {i + 1}/{args.file_num}...")
            sim_init_time = args.start_time + timedelta(hours=i * file_step * args.start_time_interval)
            if i == (args.file_num - 1):
                file_step = args.num_samples
            simulate(num_samples= args.num_samples,
                    start_time_interval=args.start_time_interval,
                    parallel=args.parallel,
                    workload_name= args.workload_name,
                    sim_init_time= args.start_time,
                    workload_len=args.workload_len,
                    warmup_len=args.warmup_len,
                    cpu_cores= args.cpu_cores,
                    output_dir=args.output_dir,
                    slurm_config=slurm_config,
                    backfill_config=backfill_config,
                    node=args.node)
            args.num_samples -= file_step
        print("-----------------------")
    else:
        simulate(num_samples= args.num_samples,
                start_time_interval=args.start_time_interval,
                parallel=args.parallel,
                workload_name= args.workload_name,
                sim_init_time= args.start_time,
                workload_len=args.workload_len,
                warmup_len=args.warmup_len,
                cpu_cores= args.cpu_cores,
                output_dir=args.output_dir,
                slurm_config=slurm_config,
                backfill_config=backfill_config,
                node=args.node)
    print("Simulation ends...")
    print("--------------------------------------------------------------------------------------------------------")

    print(f"Elapse Time: {datetime.today() - prev_time}")


if __name__ == '__main__':
    main()

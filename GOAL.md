以写完draft为目的
我们训练的PPO，为了符合故事，应该有两个特征：
1. 策略不收敛到单一选项，即不和fixed scale重合
2. 打赢fixed scale

以防你不记得：frontera是84node，ls6是88node，我们读取对应的job trace时要采用对应的node上限。

当前候选实验（2026-09-23，待在计算节点启动）：
详见 docs/WEIGHTED84_RUN.md。
84节点，动作4/16/64；alpha=0/0.2/0.5/0.8/1。
最小化 alpha*TAT/Tbar + (1-alpha)*carbon(rho=1)/Cbar，PPO奖励取负。
前5轮仅观察，不更新actor/critic；400个完整训练任务的共同均值作为参考量，随后冻结。
总64轮（5轮观察+59轮更新），重新训练PPO；复用已完成C84的fixed结果，不重跑wait probes。

当前集群为 Frontera Slurm；本地改代码，远程只 pull/运行。安装和启动见 docs/FRONTERA_RUN.md。
登录节点安装：bash scripts/frontera_env.sh
登录节点提交：sbatch scripts/frontera_weighted84.sh
中断后：sbatch scripts/frontera_weighted84.sh --resume
计算节点交互式：bash scripts/frontera_weighted84.sh [--resume]

Git 不同步 results/；本次新集群若缺旧 C84 fixed，仅重建 train/validation 三档 fixed 与参考量到 results/amsp-frontera-7b-c84-fixed-frontera，然后启动加权 PPO。不会重跑 wait probes 或旧强预算 PPO。

旧强预算C84实验已在 Sophia 完成，原结果位置 results/amsp-frontera-7b-c84-seed11；仅 pull 仓库不能把它带到 Frontera。
新实验输出 results/amsp-frontera-7b-c84-weighted-seed11。

1. 你运行在登陆节点上。永远不要运行任何heavy workload的工作
2. 当前迁移到 Frontera Slurm：本地修改代码和稿件，远程只 git pull --ff-only 与安装/启动，不在远程 commit/push 日志。需要真实实验时准备 sbatch 脚本交用户运行；同一脚本兼容计算节点交互式 bash，不在登录节点执行真实回放/训练。当前入口见 docs/FRONTERA_RUN.md。
3. 后续实验脚本将 stdout/stderr 保存到仓库 out/ 文件夹，每次运行使用独立日志文件；可同时保留终端输出。分析进度和结果时直接读取 out/ 日志及 results/ 产物，不要求用户手动粘贴日志。
4. 所有新增或修改的 Slurm 脚本统一包含 #SBATCH --mail-type=ALL 和 #SBATCH --mail-user=sf850@scarletmail.rutgers.edu。

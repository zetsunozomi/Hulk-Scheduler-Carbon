# 当前方案更新：AMSP 引用输入（2026-09-17）

当前正文用 P0=1 kW 固定参考单位，未知 workload coefficient kappa 吸收绝对整节点功率；scale coefficient 仍由 rho/eta 情景指定。保留三因子分解，但不再把任一服务器规格数值称为本项目的节点功率估计。相对 carbon 不依赖共同倍率，scale 假设仍需 E4。

目标执行平台为 AMSP 报告的八卡 A800 配置，已取消 Vista/Qwen profiling。下方内容是此前功率设计的历史说明；旧 Frontera/LS6 常数和 Vista 待测项都不是当前运行输入。

---

# 三层整节点功率模型

**平台更新（2026-09-17）**：当前测量计划改为 Vista GH 节点上的 Qwen3，每节点一张 GPU；4/8/16/32 节点即相同数量的卡。三层公式保留，Vista 的规格参考值暂留空。下方 Frontera/Lonestar6 规格表仅保留为历史来源，不能用于 Vista。GH200 是 CPU/GPU 组合模块，组合功率预算与其已包含的分量不得重复相加；辅助部件仍须单列假设。主比较使用同一 machine–workload panel 内的相对 modeled carbon，整体正系数抵消不代表规模功率假设已被验证。

我们将每个 compute node 的平均功率估计写成三个因子的乘积：

$$
\boxed{
\widehat P_{m,w,n}
=P_m^{\mathrm{spec}}\,\kappa_{m,w}\,s_{m,w,n}.
}
$$

其中 $m$ 表示机器，$w$ 表示 workload，$n\in\{4,8,16,32\}$ 表示节点规模。三个因子分别对应**规格常数、workload 系数、scale 规模系数**；后两个系数均无量纲。以四节点配置为参考，规定 $s_{m,w,4}=1$，从而区分 workload 的整体功率水平和规模变化带来的相对差异。

## 1. 规格层：由硬件组成确定的节点功率常数（下表为旧平台）

规格层采用核对后的硬件型号、数量和厂家功率规格，并保留辅助部件的估算项：

| 组成 | Frontera RTX 节点 | Lonestar6 A100 节点 |
|---|---|---|
| GPU 配置 | 4 张 Quadro RTX 5000 | 3 张 A100 PCIe 40GB |
| 单卡规格功率 | 265 W total board power | 250 W TDP |
| GPU 合计 | $4\times265=1,060$ W | $3\times250=750$ W |
| CPU 配置 | 2 颗 Xeon E5-2620 v4 | 2 颗 EPYC 7763 |
| CPU 合计 TDP | $2\times85=170$ W | $2\times280=560$ W |
| Other，估算 | 150 W | 190 W |
| **名义节点功率合计** | **1,380 W** | **1,500 W** |

Frontera 的 GPU 节点型号和数量来自 [TACC 配置说明](https://docs.tacc.utexas.edu/hpc/frontera/)。RTX 5000 采用厂家列出的 265 W 整板功率规格；E5-2620 v4 每颗 TDP 为 85 W。[NVIDIA 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf)、[Intel 规格](https://www.intel.com/content/www/us/en/products/sku/92986/intel-xeon-processor-e52620-v4-20m-cache-2-10-ghz/specifications.html)

Lonestar6 的 A100 节点配置来自 [TACC 配置说明](https://docs.tacc.utexas.edu/hpc/lonestar6/)。A100 PCIe 40GB 的规格为 250 W TDP；EPYC 7763 的默认 TDP 为 280 W。[NVIDIA 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)、[AMD 规格](https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7763.html)

Other 覆盖系统内存、主板、本地存储、网卡、风扇和节点内供电损耗。**150/190 W 是作者估算，GPU 和 CPU 的规格功率也不等于训练时的平均功率。** 因而，本层定义的是名义节点功率参考值，不是整节点实测值，也不是经过验证的整节点功率上限。

采用未四舍五入的合计作为模型常数：

$$
P_m^{\mathrm{spec}}=
\begin{cases}
1.38\text{ kW},&\text{Frontera RTX},\\
1.50\text{ kW},&\text{Lonestar6 A100}.
\end{cases}
$$

这个常数由机器配置决定，不随 workload、节点规模或调度方法改变。

## 2. Workload 层：训练负载对应的功率系数

不同 workload 的计算、访存和主机活动不同，即使运行在同一种节点上，平均功率也不一定相同。用 $\kappa_{m,w}>0$ 表示某一 workload 在该机器上的参考功率相对于规格常数的比例：

$$
\widehat P_{m,w,4}
=P_m^{\mathrm{spec}}\,\kappa_{m,w}.
$$

本层将比较规模固定为四节点。因此，$\kappa_{m,w}$ 只描述机器和 workload 对应的整体功率水平，不随动作 $n$ 改变；规模变化由第三层单独处理。

在没有整节点遥测时，$\kappa$ 是估计或敏感性参数。单卡测量可以为 GPU 分量提供证据，但不能单独确定完整节点的 $\kappa$，也不能据此声称整节点功率已经实测校准。

同一个 machine–workload panel 内，所有 fixed baseline 和 adaptive policy 共用相同的 $\kappa$。保持规模系数不变时，单独调整 $\kappa$ 只会整体缩放该 panel 的碳排坐标，不会改变支配关系。

## 3. Scale 层：节点规模对应的相对功率系数

用 $s_{m,w,n}$ 表示相同 workload 在规模 $n$ 下的平均每节点功率，相对于四节点配置的比例：

$$
s_{m,w,n}
=\frac{\widehat P_{m,w,n}}{\widehat P_{m,w,4}},
\qquad s_{m,w,4}=1.
$$

利用现有、完成同样训练工作的 scaling profile，定义：

$$
W_{m,w,n}=nT_{m,w,n},
\qquad
\eta_{m,w,n}=\frac{W_{m,w,4}}{W_{m,w,n}}.
$$

其中 $T_{m,w,n}$ 是完成相同参考更新数的 active runtime，$W_{m,w,n}$ 是对应的 node-hours。若规模增加时累计 node-hours 增多，则 $\eta_n$ 下降；始终有 $\eta_4=1$。新的 Vista 曲线尚未测量，不能预写其趋势。

将规模系数具体构造为：

$$
\boxed{
s_{m,w,n}
=\rho_{m,w}+(1-\rho_{m,w})\eta_{m,w,n},
\qquad 0\le\rho_{m,w}\le1.
}
$$

$\rho$ 控制规模系数的变化幅度：$\rho$ 对应参考功率中不随有效计算占比同比下降的份额，$1-\rho$ 对应随有效计算强度变化的份额。**$\rho$ 不等于 CPU＋Other 的部件占比，$\eta_n$ 也不是实测利用率。** 这一构造是扩展效率驱动的功率情景模型，假设完成相同有效训练工作的计算动态能量近似不变。

| 参数设置 | 规模系数的含义 |
|---|---|
| $\rho=1$ | $s_n=1$，每节点功率与规模无关 |
| $0<\rho<1$ | 扩展效率下降时，每节点功率也有所下降，但保留基础功率部分 |
| $\rho=0$ | $s_n=\eta_n$，在本模型内，相同有效工作的训练总能量相同；属于理想极限 |

可以用 $\rho\in\{0.25,0.5,0.75,1\}$ 进行情景扫描。这些取值是实验设计参数，不是实测范围或置信区间。

将三个层次合并，得到最终模型：

$$
\boxed{
\widehat P_{m,w,n}
=P_m^{\mathrm{spec}}\,
\kappa_{m,w}\,
\left[\rho_{m,w}+(1-\rho_{m,w})\frac{W_{m,w,4}}{W_{m,w,n}}\right].
}
$$

同一机器、workload 和节点规模下，所有方法使用相同的功率估计；策略之间的差异来自其实际选择的节点规模及执行时段。

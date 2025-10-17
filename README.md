# SymMaP
SymMaP: Improving Computational Efficiency in Linear Solvers through Symbolic Preconditioning
├─ README.md
├─ requirements.txt       # 依赖环境清单
├─ LICENSE                # 开源协议文件
├─ src/                   # 核心代码文件夹
│  ├─ train.py            # 训练脚本
│  ├─ evaluate.py         # 评估/测试脚本
│  └─ model/              # 模型定义文件夹
│     └─ my_model.py      # 自定义模型代码
├─ data/                  # 数据相关（可选，或放数据获取说明）
│  └─ precondition        # 数据下载链接/处理步骤
│     ├─ e.c              # PETSc方程求解文件
│     ├─ makefile         # cmake文件
│     └─ generate.ipynb   # 创建数据集核心文件
└─ results/               # 实验结果（可选，放关键结果日志或可视化）

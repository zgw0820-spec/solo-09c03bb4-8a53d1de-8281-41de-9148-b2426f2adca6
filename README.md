# 风机停机预警训练平台（可续跑）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `worker-1-replacement` | 可选（`profiles: ["replacement"]`）的**同名**替换容器，用于演示滚动替换 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3}`，返回 201 与任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重；完成时含 `final_weights` |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、完成、丢失租约 |
| `GET` | `/workers` | 工作器实例（`instance_id`）、心跳与租约持有者（含实例号） |
| `GET` | `/health` | 健康检查 |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

### 同名实例接替（滚动替换）

工作器角色名（`worker-1`）是运维标识，**不是进程身份**。每个进程启动时还会生成随机
`instance_id`，所有租约授权都校验三元组 `(worker_id, instance_id, epoch)`：

1. `workers` 表按角色名只保存**当前**注册的实例。新进程以同名注册时，旧进程立即被
   "退休"：它的心跳、续约、断点提交全部失败（`heartbeat`/`renew` 返回 `False`，
   `advance_job` 抛 `LeaseLost`），因此旧心跳无法覆盖新实例的状态。
2. 同名接替**不等待租约超时**：新实例领取任务时直接分配从未复用的新 `epoch`，旧实例
   立刻被围栏挡住。异名工作器仍然必须等持有者心跳超过 `lease_seconds` 才能接管。
3. `release` 的 `DELETE` 同时匹配 `worker_id`、`instance_id` 和 `epoch`，旧进程退出时
   删不掉新进程的租约。
4. 新实例随后也故障（崩溃、不释放租约）时，异名工作器在租约超时后从**最近确认的
   断点**续训；由于每个断点都原子提交权重/动量/进度，最终权重与不中断训练逐位一致。
5. 数据库升级：`init_db` 会为旧卷中的 `workers`/`leases`/`checkpoints` 自动补
   `instance_id` 列（旧行取空串，注册后即被新实例取代），无需手动迁移。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的四项自检（含同名接替）
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 多个工作器进程、SIGKILL 接管、同名接替、迟到工作器围栏）
python3 -m pytest -q
```

同名接替可直接复现（API 需已启动，且与工作器共用同一个 `TRAINING_DB_PATH`）：

```bash
# 终端 A：观察进度
curl -s localhost:8000/jobs/<job_id> | python3 -m json.tool

# 终端 B：同名滚动替换 + 新实例再次故障 + 异名接管
export TRAINING_DB_PATH=/path/to/training.db   # 与 API/Compose 卷中的同一文件
python3 scripts/replacement_demo.py --api http://127.0.0.1:8000
```

Docker Compose 下的等价操作：

```bash
docker compose up -d --build
# 在 worker-1 正在训练时，再起一个同名容器（旧容器仍在运行也不会再抢占）
docker compose run --rm --no-deps worker-1-replacement
# 想模拟替换者随后也故障：在另一个终端 docker kill 该容器（或 Ctrl-C）
# 之后 worker-2 在租约超时后接管，从最近确认断点续训
```

演示脚本第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger
app/worker.py     工作器主循环、租约续训
app/store.py      任务状态机、批次提交、断点查询
app/lease.py      租约获取/续约/实例围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验与标准化
app/db.py         SQLite 连接与表结构（含旧卷列迁移）
tests/            单元、API、进程级集成测试（含同名接替）
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/replacement_demo.py  同名滚动替换 + 故障续训现场演示
scripts/selftest.py  无 Docker 自检
```

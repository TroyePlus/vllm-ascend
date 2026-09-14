# 131 A2 验证记录：2026-09-14

容器：`8.92.7.131 / dsv4-pr72-final`。测试均在 131，不占用 127 NPU。

## 固定环境

- vLLM：`/vllm-workspace/vllm`，Git `f5ffef0859241759035b4cc3db0360b9c1d41048`。
- torch `2.10.0+cpu`，torch_npu `2.10.0.post2`。
- 已安装 FXRT `0.1.dev0+5a29416`，本任务没有重新安装、更改 site-packages 或编译 FXRT。
- 已安装 `fxrt/torch/fx_backend.py` SHA256：
  `a099e1a852b3018276f61e7ca680d77d5056443b62efcd112293121532cae0e2`。
  与 Git `5a29416` 对应文件一致。
- 已安装 `libops_ascend_aclnn.so` SHA256：
  `0b7311baae4ec8d061ace3cf1b46957a04a904bd06ae2b3305eb048e3beb7c91`。
- `/workspace/fxrt` 存在之前未提交的 fx_backend.py/symbolic_shape.py/utils.py 修改，
  本次未使用这些工作区修改编包；不能将该目录状态当作已安装 wheel 的内容。
- 三个 Ascend worktree 共用之前从基线构建的 C++ 扩展（显式 symlink）：
  `/workspace/variants/opaque/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`，
  SHA256 `7c6e4b62c000730928e5463b9dd4dce9b6313de68215da02f02913bedd19b658`。
  另共用此目录下的 kernel library/vendors；两个基线之间 csrc diff 为空。

## 已完成的测试

- 三分支分别运行 `tests/ut/test_moe_route_audit.py`：每次 3 个测试通过。
  覆盖采样限流、关闭开关、异常原样抛出、不读取 Tensor 值、图外 CPU profiling +
  `torch.compile(fullgraph=True)` 两次调用只编一次。
- 三分支分别运行 `tests/ut/test_moe_audit_filter.py`：每次 2 个测试通过。
  覆盖多 rank 合并、缺失 END、损坏记录、无日志显式报错。
- 三分支运行 `tools/moe_audit/check_selectors.py`：直接抽取对应版本实际 selector 函数，
  模拟 EP8、256专家、capacity32、fused0，检查32/33/256/2048/8192。
  A2 全部 ALLGATHER；A3 为32→MC2，其余→ALLTOALL。
  **这是 A3 选择器单测，不是 A3 通信 kernel 实测。**
- 变更的 Python helper/test/env/context 文件 Ruff check 通过，shell `bash -n` 通过。
  `bash format.sh ci` 未能运行：环境未安装 pre-commit，不能声称全仓 lint 通过。

## 八卡服务测试

DP2、TP4、EP8、DSA-CP、shared expert DP、MTP1、max_len4096、batch8192、seqs16。
测试只使用裁剪4层 dummy 模型；显存比例0.8、本机网络/RPC端口，不是生产真实权重性能验收。
配置请求 CPU binding=true，但日志报告 `Bind cpus failed ... Skip binding cpu`。
HTTP 请求直接访问 P API，max_tokens=1；不是完整现网8P+8D/proxy的10-token响应测试。

| 模式 | 运行时 Ascend commit | 请求 | 结果 |
| --- | --- | --- | --- |
| eager | `fdabe9be2` | 256、256、2048、2048 | 四次 HTTP200，usage核实输入长度 |
| opaque FXRT | `060f2ccbe` | 256、256、2048、2048 | 四次 HTTP200，存在外部backend入口及FX图 |
| split FXRT | `d57d7f385` | 256、256、2048、2048 | 重试四次 HTTP200，存在外部backend入口及展开的FX图 |

后续文档、过滤脚本、格式化提交未改变上述日志采样语义。
三轮所有采样 SELECT 均为 ALLGATHER、capacity32；启动8192是profile，不是用户请求。
OPS 同时可见 DSA-CP `c10d::alltoall_base_` 和 MoE `npu_moe_init_routing_custom`，
说明仅搜索 AllToAll 字符串不足以判断 MoE 路由。
opaque 测试停机时向进程组发 TERM 产生 SystemExit 清理日志；四个请求均已返回后发生，
后续测试驱动改为先停止 API、等待退出，再清理本次 session 的残留进程。

拆分版首次启动在设备5初始化失败：`507033 / E39007 SetDevice`，发生在模型加载前。
保留日志：`/workspace/dsv4/logs/7323-split-device-init-failure/`；已终止其残留进程。
重试在02:29–02:30 UTC完成四次请求，API正常退出；再次检查 NPU0–7 均无运行进程。
三轮过滤脚本均成功解析全rank，malformed=0。Eager共112组采样，opaque/split各128组，
每组都有SELECT、OPS、END。回传摘要约8–12KB（根据模式与输入signature不同而变化）。
split没有人为强制A2走AllToAll，因此没有复现A3的async_op=True异常；这不等于A3路径已修复。

## 可追踪位置

- 三个源码：`/workspace/variants/audit7323-{eager,opaque,split}`，均有完整 Git 信息。
- 运行日志：`/workspace/dsv4/logs/7323-{eager,opaque,split}/rank{0,1}.log`。
- FX 图：相应日志目录的 `fx_dump/`。
- 过滤结果：相应日志目录 `prefill.audit.txt`，仅需回传此类摘要。
- 本次测试驱动：`/workspace/dsv4/validate_7323.py`；宿主留档
  `/home/chenhaozhe/liyizhan/validate_131.py`，127副本
  `/home/liyizhan/dsv4/audit7323/validate_131.py`。
- 本次未修改127主工作区既有实验改动，也未改 vLLM/FXRT 代码。

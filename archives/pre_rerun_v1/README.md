# 旧实验归档：pre_rerun_v1

本目录保存统一重跑前的全部历史结果，不再作为新训练的输出位置。

- `outputs/`：原outputs完整搬入，内部目录与文件内容保持不变。
- `source_snapshot/`：本次整理前的全部Python源码，以及旧src说明、服务器指南、依赖文件和gitignore。用于追溯代码版本；不会自动用于新训练。
- `server_logs/`：原项目根目录旧CRNN日志。
- `archive_manifest.json`：逐文件SHA256和新旧源码路径映射。搬迁时212个原输出文件全部逐字节校验一致。

注意：旧config和旧摘要内的绝对路径/`outputs/...`字符串按历史原样保留，并非当前新结果地址。从本目录下的 `outputs/` 查旧run即可。不要修改这些原始配置来冒充新运行。现有截图但尚未同步的100轮结果不在此归档中。

新版入口见项目根目录 `RERUN_SERVER.md`；新实验输出位于项目 `outputs/rerun_v1/`。

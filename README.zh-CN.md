# Pigeon Score Scan

[English](README.md)

Pigeon Score Scan 是一款本地运行的 Windows 桌面程序，用于将方向正确的西方五线谱印刷扫描件转换为 MXL 和 MusicXML。识别使用 CPU，导入文件和处理结果保留在运行程序的电脑中。

> **项目状态：** 0.37 开发快照。源码和测试已公开，但目前不宣称稳定版或生产准确度保证。

## 支持范围

- 单谱表器乐谱。
- 常见钢琴谱，包括同一键盘谱表内的独立声部和跨谱表记谱。
- 纵向排列的多个单声部器乐声部；各乐器不要求音符横向对齐。
- 钢琴与多个单声部器乐声部的组合谱。
- 方向正确的图片扫描件和 PDF；连续页面按乐谱顺序导入。

当前验证目标为每个谱表系统最多 16 个物理谱表。键盘声部可包含临时附加谱表或 ossia。完整边界见[场景与功能限制](docs/SCOPE.zh-CN.md)。

暂不支持：手写谱、带透视畸变的照片、打击乐谱、TAB、歌词、和弦符号、数字低音、浓缩或 divisi 谱表、真正的复合节拍体系、微分音或图形谱，以及把分别扫描的独立分谱重建为一份总谱。

## 输出文件

- MXL，可使用 MuseScore 查看和编辑。
- 未压缩的 MusicXML。
- 本地预览和机器可读的转换报告。

若结构或完整性检查没有全部通过，程序会明确标记需要人工复查，不会把结果静默标记为已验证。

## 隐私与安全

桌面服务仅监听 `127.0.0.1`，本地请求需要访问令牌。扫描件、中间文件和结果保存在便携工作区中。不要将服务端口暴露到网络。

安全问题请通过 [GitHub 私密漏洞报告](SECURITY.md)提交。除非拥有分享许可，否则不要附带受版权保护的乐谱。

## 开发环境

需要 Windows 10/11、Python 3.12 或 3.13，以及 uv 0.9.26。PowerShell：

```powershell
git clone https://github.com/KalePotato/Pigeon-Score-Scan.git
Set-Location Pigeon-Score-Scan

py -3.12 -m pip install uv==0.9.26
uv sync --project app --locked --group dev
$env:PYTHONPATH = (Resolve-Path "app/src").Path

uv run --project app python -m scorescan --self-test --json --root .
uv run --project app python -m pytest -q app/tests
node --check app/src/scorescan/web/app.js
```

开发模式启动：

```powershell
$env:PYTHONPATH = (Resolve-Path "app/src").Path
uv run --project app python -m scorescan
```

Windows 便携包由发布流程构建，不提交到源码仓库。构建说明见 [BUILDING.md](BUILDING.md)。

2026-08-02 的 Windows 公开源码验证通过 1,382 项测试。另有 10 项外部语料集成测试按预期跳过；这些可选数据集不随仓库分发。CI 会使用提交的源码和锁文件重新执行同一套公开测试。

贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。项目使用 [GNU AGPL v3 或更高版本](LICENSE)。第三方组件的许可见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) 和 `licenses/`。

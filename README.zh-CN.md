# Pigeon Score Scan

[English](README.md)

Pigeon Score Scan 是一款本地运行的 Windows 乐谱识别程序，用于将印刷五线谱扫描件转换为 MXL 和 MusicXML。识别使用 CPU；扫描件和输出文件保留在运行程序的电脑中。

## 下载

从 [Releases](https://github.com/KalePotato/Pigeon-Score-Scan/releases) 下载 Windows x64 压缩包，完整解压后运行 `pigeon-score-scan.exe`。

首次启动需要联网安装已锁定版本的 Python 运行时、依赖和识别模型，后续启动使用本地便携运行时。当前提供的是未签名的开发预发布版，Windows 可能显示安全提示。

## 支持范围

- 单谱表器乐谱。
- 常见钢琴谱，包括独立声部和跨谱表记谱。
- 纵向排列的多个单声部器乐声部，各乐器不要求音符逐拍横向对齐。
- 钢琴与多个单声部器乐声部的组合谱。
- 方向正确并按乐谱顺序导入的图片扫描件和 PDF。

当前验证边界为每个谱表系统最多 16 个物理谱表。手写谱、带透视畸变的照片、打击乐谱、TAB、歌词、和弦符号、数字低音、浓缩或 divisi 谱表、真正的复合节拍体系、微分音和图形谱不在当前范围内。完整说明见[场景与功能限制](docs/SCOPE.zh-CN.md)。

## 输出文件

程序输出可在 MuseScore 中查看和编辑的 MXL、未压缩的 MusicXML、本地预览和转换报告。未通过输出检查的结果会标记为需要人工复查。

## 开发

支持 Windows 10/11 和 Python 3.12/3.13。环境准备、测试和发布命令见 [BUILDING.md](BUILDING.md)，贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

安全问题请使用 [GitHub 私密漏洞报告](SECURITY.md)。

## 许可证

Pigeon Score Scan 使用 [GNU AGPL v3 或更高版本](LICENSE)。第三方组件保留各自许可证，详见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) 和 `licenses/`。

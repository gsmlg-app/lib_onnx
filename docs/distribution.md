# 原生构建与消费端验证（ABI 2 候选版本）

`0.2.0-dev.1` 使用 Dart native-assets build hooks；核心包不依赖 Flutter engine。
最低 Dart 为 3.10，实际验证工具链见本轮 remediation 记录（Dart 3.13.7 / Flutter
3.47.9）。普通消费者无需 CWD、Homebrew、工作区 path dependency 或手动复制 dylib。

## 默认路线

Pub 包包含 Dart API、C bridge 源码和锁定的 ORT 1.31.1 头文件；不包含开发机原生库。
`hook/build.dart` 从 `native/ort_artifacts.json` 选择**精确目标**的官方发行档案，验证固定
SHA-256，编译 bridge，再把 bridge 和 ORT 交给 SDK 作为 code assets 打包。首次构建需要
C 工具链、tar 和网络；后续复用校验过的归档缓存。加载时平台 loader 校验架构，Dart loader
校验 bridge ABI = 2、ORT = 1.31.1；旧 bridge 或不匹配 ORT 被明确拒绝。

| 目标 | 默认构建输入 | 本轮声明 |
|---|---|---|
| macOS arm64、macOS 14+ | 官方 `onnxruntime-osx-arm64-1.31.1.tgz`，CoreML/CPU | 候选；实际构件/consumer 证据与硬件验收分开记录 |
| Linux x64/arm64 | 对应官方 Linux 1.31.1 归档 | 仅 CPU 正确性/安全测试路线，不是 Linux GPU 支持 |
| iOS 真机 arm64 | 需要单独 iPhoneOS ORT/CoreML 构件 | experimental；未提供构件时明确拒绝构建 |
| iOS 模拟器 | 需要单独 iPhoneSimulator 构件 | experimental；不能以 macOS dylib 或真机构件替代 |
| Android arm64-v8a | ORT QNN + 合法取得的匹配 HTP 依赖 | experimental；SDK/构件/最终 APK 验收尚需独立完成 |

`GetAvailableProviders` 仅表示构件编入的 EP。已检查的官方 ORT dylib 的 `LC_BUILD_VERSION` 明确要求 macOS14.0；
Dart JIT 的通用 hook target 默认12 不会改变实际最低系统。消费项目的 macOS deployment
target 应设为14.0。默认 macOS ORT 还可能列出 WebGpu；这不表示
本库已实现或验证该路线。Session 注册 CoreML 也不证明本次运行使用 GPU/ANE。

## 离线缓存与显式构件

在**消费项目**的 `pubspec.yaml` 配置受 build-hook 协议跟踪的输入，而不是进程 CWD：

```yaml
hooks:
  user_defines:
    lib_onnx:
      ort_archive: /absolute/cache/onnxruntime-osx-arm64-1.31.1.tgz
      offline: true
```

也可设置 `ort_cache: /absolute/cache/`，其中归档文件名必须与 lock 一致。所有本地归档仍按
固定 SHA-256 校验；缺失或校验失败直接报错，不下载其他版本、不退回系统 ORT。
网络需要代理时应优先预取官方归档，再用离线输入；下载器保留 TLS 验证并尊重可用的标准
代理环境。缓存不是无需验证的信任来源。

开发者仍可显式 `OnnxRuntime.create(libraryPath: '/absolute/bridge')`。该功能要求自行提供
兼容的 transitive dependencies；同样校验 ABI/ORT，不参与普通消费者搜索，也不绕过
平台 loader 的架构校验。

对 iOS/Android 或自建 ORT，设置 `native_bundle: /absolute/verified-bundle/`。目录内必须
包含 `artifacts_manifest.json`，示例结构如下（散列必须来自**实际文件**，不可使用占位值）：

```text
manifest_version: 2
bridge_abi_version: 2
ort_version: "1.31.1"
target: ios_arm64 | android_arm64
# iOS 必须额外写 ios_sdk: iphoneos 或 iphonesimulator
# Android 必须写 page_size_check: passed，并接受 hook 对每个 ELF 的实际检查
binaries:
  - name: 目标 bridge 文件名
    role: bridge
    sha256: 实际文件的 64 位 SHA256
  - name: 目标 ORT 文件名
    role: runtime
    sha256: 实际文件的 64 位 SHA256
  - name: 依法取得且确实需要打包的 HTP 依赖文件名
    role: qnn_htp_dependency
    sha256: 实际文件的 64 位 SHA256
```

文件使用 JSON 编码。hook 要求恰好一个 bridge 和对应 runtime，验证目标、SDK、版本、校验和；
SDK native-assets 校验器检查 Mach-O/ELF 目标。Android 对每个提供的库额外读取真实 ELF
PT_LOAD，检查 AArch64 与 16 KB alignment。manifest 的 EP 声明不能代替运行时能力查询或
设备初始化证据。自建包的许可证、签名和 transitive dependency 检查仍需实际完成。

## 平台构建与审核

`native/CMakeLists.txt` 接受 `ORT_LIBRARY` 和 `ORT_INCLUDE_DIR`；跨平台必须显式提供目标库。
iOS 使用对应 Xcode SDK/toolchain 和 device/simulator 构件；不会在 `APPLE` 分支统一链接
macOS dylib。macOS 本地 `make -C native` 是开发/原生 sanitizer 路线，普通消费者走 hooks。

Android 目标为 NDK **28.2.13676358**、`arm64-v8a`，默认不上传或自动取得 Qualcomm SDK。
提供 ORT QNN HTP 构件、匹配 SoC 的 HTP stub/skel/系统依赖及其合法分发依据；HTP backend
使用 `libQnnHtp.so`，不得以 QNN CPU/GPU backend 冒充。链接 `max-page-size=16384` 只作用于
bridge，不能证明 ORT、QNN 或最终包兼容。实际产物必须运行：

```sh
python3 tools/audit_android_pages.py path/to/libbridge.so path/to/libonnxruntime.so path/to/libQnnHtp.so
python3 tools/audit_android_pages.py path/to/app-release.apk path/to/app-release.aab
# 使用实际安装的 Android build-tools；AAB 还需检查 bundletool 生成的最终 APK。
zipalign -c -P 16 -v 4 path/to/app-release.apk
```

工具会按 magic 检查归档内的**每个** ELF（包括 `.so` 以外的 DSP 文件），不会只检查 bridge。最终 APK 未压缩库还需 16 KB ZIP
对齐；AAB 内 ELF 通过不代表交付 APK 通过。没有 APK/AAB 时只能记为 not-run。

## 干净消费者与构件收据

```sh
# 使用 Pub 自身的 --to-archive 创建实际待发布 payload，启动临时本地 hosted registry。
# 消费项目位于仓库外、使用全新 Pub cache，无 path dependency 或预拷贝原生库。
python3 tools/verify_consumer.py --output build/consumer-evidence \
  --ort-archive /absolute/cache/onnxruntime-osx-arm64-1.31.1.tgz --flutter

# 从实际编译的 bridge 查询 ABI/ORT/构件 EP，并计算真实文件校验和。
python3 tools/generate_artifacts_manifest.py \
  --bridge native/build/liblib_onnx_bridge.dylib \
  --runtime native/third_party/onnxruntime/lib/macos_arm64/libonnxruntime.dylib \
  --target macos_arm64 --evidence docs/reviews/runtime-acceleration-remediation.md
```

consumer 脚本执行真实 CPU Identity ORT 模型，验证 JIT 和 release CLI 在无关 CWD 启动；
`--flutter` 还以进程局部 `FLUTTER_MACOS_ARM64_ONLY=true` 构建并启动 macOS arm64 release app，
记录 bundle、签名、transitive dependencies；不会修改用户的全局 Flutter 配置。
Flutter 默认 universal release 会显式拒绝未支持的 macOS x64 目标。
本地 `codesign` 验证仅证明所检查签名有效；ad hoc 签名不代表 Developer ID、notarization
或商店分发通过。这些测试只证明该小模型/交付链，不证明 Qwen 或 ANE。`commands.log`、`receipt.json` 和
产物保存在指定 output；原始 Dart/Flutter 输出不被模拟成功路径替代。

hook 为实际生成的文件写构件收据；仓库根 manifest 由显式脚本生成。配置、编译、打包、
干净 consumer、设备测试分别标记。source fingerprint 防止源码变更后沿用旧验收状态。
ORT MIT license 和 ThirdPartyNotices 随固定归档保留，最终 app 分发也必须携带相关通知。

## 发布门禁与历史更正

过去 v0.1 文档的“所有平台产物已就绪”“16 KB 已通过”和统一自动搜索不是本轮可复用的
验收证据；历史 Git 提交保持不变。本轮新 ABI 使用 `0.2.0-dev.1`，不改写 v0.1 tag。

`release.yml`、`publish.yml`、`first-publish.yml` 在发布动作前调用
`python3 tools/release_gate.py`。软件测试必须有证据；stable 额外要求每个显式声明的平台
编译、打包、干净消费者、设备验收通过，并有真实 binary hashes。宣称加速还需硬件观测。
未完成的平台不列为 production-ready；仅跳过真机测试不能使门禁变绿。候选 GitHub release
使用 prerelease 标识；push 本身只触发 CI，不发布 stable。本任务不发布新 stable。

Pub 的 exact-pin、`docs/`/`tools/` 命名及工作区修改等 warning 保留于日志；独立 analyzer、
原生回归与消费端测试必须通过。`pub publish --dry-run` 不能替代上述原生集成验收。

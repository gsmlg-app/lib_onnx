# 构件分发与部署规范

## 1. 分发架构与包结构

`lib_onnx` 遵循确定性、防篡改的可重复构建与分发规范：
1. **Pub Package (`lib_onnx`)**：
   - 包含纯 Dart API、FFI 绑定、运行时管理、Session 抽象、张量容器及 `EmbeddingAdapter`；
   - 包含原生 C ABI 头文件 `lib_onnx.h` 及构建脚本；
   - **不内置任何模型权重文件**，保证 pub 包体积轻巧稳定。
2. **原生二进制构件（Prebuilt Native Binaries）**：
   - 包含各目标平台的 `liblib_onnx_bridge` 及对应版本的 `onnxruntime` 动态库；
   - 发布在官方 GitHub Release 版本构件库中，受 `artifacts_manifest.json` 的 SHA-256 校验和保护。

---

## 2. 离线构建与本地构件覆盖

企业及受控私有网络环境支持完全离线构建与运行：
- **环境变量 / 路径覆盖**：可在应用启动时调用 `LibOnnxBindings.init(libraryPath: '/path/to/custom/liblib_onnx_bridge.dylib')` 指向本地预置的动态库。
- **自动搜索优先级**：
  1. 显式指定的自定义绝对路径；
  2. 应用程序运行目录及 `native/build/`；
  3. 应用沙盒的 `Frameworks/`（macOS / iOS）或原生库目录（Android `lib/arm64-v8a/`）；
  4. 系统全局库路径。

---

## 3. 第三方许可与发布限制

- ONNX Runtime 遵循 MIT 许可协议；
- Apple CoreML 依赖操作系统内置系统框架；
- Qualcomm QNN 运行库由 Qualcomm 授权分发，对于需要预置 HTP 驱动的应用，开发者需按照高通 SDK 许可打包 `libQnnHtp.so`，`lib_onnx` 提供透明的 `qnnBackendPath` 配置接口。

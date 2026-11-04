import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lib_onnx/lib_onnx.dart';

void main() => runApp(const LibOnnxExampleApp());

class LibOnnxExampleApp extends StatelessWidget {
  const LibOnnxExampleApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'lib_onnx native inference',
        theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
        home: const HomeScreen(),
      );
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  OnnxRuntime? _runtime;
  InferenceSession? _session;
  ExecutionProvider _provider = switch (defaultTargetPlatform) {
    TargetPlatform.macOS || TargetPlatform.iOS => ExecutionProvider.coreML,
    TargetPlatform.android => ExecutionProvider.qnn,
    _ => ExecutionProvider.cpu,
  };
  bool _allowCpuFallback = false;
  bool _running = false;
  String _status = 'Ready';
  String _runtimeInfo = 'Not initialized';
  DiagnosticReport? _report;
  String? _json;
  List<double>? _output;
  List<String> _compiledProviders = const [];

  @override
  void initState() {
    super.initState();
    unawaited(_initialize());
  }

  Future<void> _initialize() async {
    try {
      final runtime = await OnnxRuntime.createAsync();
      if (!mounted) {
        await runtime.close();
        return;
      }
      setState(() {
        _runtime = runtime;
        _compiledProviders = runtime.getAvailableProviders();
        _runtimeInfo =
            'Bridge ${OnnxRuntime.bridgeVersion}; ORT ${OnnxRuntime.ortVersion}';
      });
    } catch (error) {
      if (mounted)
        setState(() {
          _status = 'Native runtime unavailable: $error';
        });
    }
  }

  @override
  void dispose() {
    // close() stops accepting work, cancels/drains accepted tasks, then frees.
    unawaited(_runtime?.close());
    super.dispose();
  }

  Future<void> _run() async {
    final runtime = _runtime;
    if (runtime == null || _running) return;
    setState(() {
      _running = true;
      _status = 'Loading the bundled MLP model…';
      _report = null;
      _json = null;
      _output = null;
    });
    OnnxTensor? input;
    Map<String, OnnxTensor>? outputs;
    try {
      final asset = await rootBundle.load('assets/mlp_smoke.onnx');
      final session = await InferenceSession.loadBytes(
        asset.buffer.asUint8List(asset.offsetInBytes, asset.lengthInBytes),
        runtime: runtime,
        options: SessionOptions(
          policy: ExecutionPolicy(
            providerPriority: [
              _provider,
              if (_allowCpuFallback && _provider != ExecutionProvider.cpu)
                ExecutionProvider.cpu,
            ],
            fallbackPolicy: _allowCpuFallback
                ? FallbackPolicy.allowed
                : FallbackPolicy.strict,
            // This is an acceleration preference. A registered EP alone does
            // not establish physical GPU/NPU use.
            hardwareRequirement: HardwareRequirement.any,
          ),
        ),
      );
      _session = session;
      if (mounted) setState(() => _status = 'Running native ONNX inference…');
      input =
          OnnxTensor.fromFloat32([1, 4], Float32List.fromList([-1, 0, 1, 2]));
      final result = await session.runWithDiagnostics({'input': input});
      outputs = result.outputs;
      final values = outputs['output']!.asFloat32List().toList();
      const expected = [0.0, 0.5, 1.5, 2.5];
      for (var i = 0; i < expected.length; i++) {
        if (!values[i].isFinite || (values[i] - expected[i]).abs() > 1e-5) {
          throw StateError(
              'MLP result mismatch: expected $expected, got $values');
        }
      }
      if (mounted) {
        setState(() {
          _output = values;
          _report = result.diagnostic;
          _status = 'Native MLP result verified. This does not validate Qwen.';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _status = 'Inference failed: $error');
    } finally {
      input?.dispose();
      for (final tensor in outputs?.values ?? <OnnxTensor>[]) {
        tensor.dispose();
      }
      await _session?.close();
      _session = null;
      if (mounted) setState(() => _running = false);
    }
  }

  void _exportReport() {
    final report = _report;
    if (report == null) return;
    setState(() {
      _json = const JsonEncoder.withIndent('  ').convert({
        'scope': 'bundled mlp_smoke.onnx inference only; not Qwen validation',
        'requested_provider': _provider.ortName,
        'active_provider': report.activeProvider?.ortName,
        'evidence': report.evidenceLevel.description,
        'duration_us': report.durationUs,
        'model_hash': report.modelHash,
        'hardware_details': report.hardwareDetails,
        'fallback_occurred': report.fallbackOccurred,
        'node_distribution': report.nodeDistribution,
        'output': _output,
      });
    });
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('lib_onnx native inference')),
        body: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(_runtimeInfo),
            Text('EPs in this ORT build: ${_compiledProviders.join(', ')}'),
            const Text(
                'Build capability does not prove device initialization or GPU/NPU use.'),
            const SizedBox(height: 20),
            DropdownButtonFormField<ExecutionProvider>(
              initialValue: _provider,
              decoration: const InputDecoration(labelText: 'Requested backend'),
              items: const [
                DropdownMenuItem(
                    value: ExecutionProvider.coreML,
                    child: Text('CoreML acceleration preference')),
                DropdownMenuItem(
                    value: ExecutionProvider.qnn,
                    child: Text('QNN HTP acceleration preference')),
                DropdownMenuItem(
                    value: ExecutionProvider.cpu,
                    child: Text('Explicit CPU correctness baseline')),
              ],
              onChanged: _running
                  ? null
                  : (value) {
                      if (value != null) setState(() => _provider = value);
                    },
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Allow CPU fallback'),
              subtitle: const Text('Strict rejects ORT CPU graph fallback. '
                  'It does not prohibit CPU inside CoreML.'),
              value: _allowCpuFallback,
              onChanged: _running
                  ? null
                  : (value) => setState(() => _allowCpuFallback = value),
            ),
            const SizedBox(height: 12),
            FilledButton(
              onPressed: _runtime == null || _running ? null : _run,
              child: Text(_running ? 'Running…' : 'Run bundled ONNX MLP'),
            ),
            const SizedBox(height: 20),
            SelectableText(_status),
            if (_output != null)
              SelectableText('relu([-1, 0, 1, 2] + 0.5) = $_output'),
            if (_report case final report?) ...[
              const Divider(),
              Text(
                  'Actual provider: ${report.activeProvider?.ortName ?? 'unknown'}'),
              Text('Evidence: ${report.evidenceLevel.description}'),
              Text(
                  'Native duration: ${report.durationMs.toStringAsFixed(3)} ms'),
              SelectableText(
                  'Model SHA-256: ${report.modelHash ?? 'unavailable'}'),
              Text(
                  'CPU fallback: ${report.fallbackOccurred?.toString() ?? 'unknown'}'),
              Text(report.hardwareDetails),
              OutlinedButton(
                  onPressed: _exportReport,
                  child: const Text('Show actual diagnostic JSON')),
            ],
            if (_json != null) SelectableText(_json!),
          ],
        ),
      );
}

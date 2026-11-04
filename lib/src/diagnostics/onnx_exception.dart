/// A failure retains its own native error; unrelated requests cannot overwrite it.
class OnnxException extends StateError {
  final int code, ortCode;
  final String stage;
  final int? taskId, requestId;
  final List<String> providers;
  OnnxException(
    super.message, {
    required this.code,
    this.ortCode = 0,
    required this.stage,
    this.taskId,
    this.requestId,
    this.providers = const [],
  });
  @override
  String toString() =>
      'OnnxException(code: $code, ort: $ortCode, stage: $stage, '
      'task: $taskId, request: $requestId, providers: $providers): $message';
}

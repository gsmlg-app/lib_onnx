import 'dart:async';

/// Options for an individual inference execution run.
class RunOptions {
  /// Unique request identifier for logging, diagnostics, and cancellation.
  final int requestId;

  /// Optional execution timeout.
  final Duration? timeout;

  /// Optional cancellation token allowing cooperative cancellation of this run.
  final CancellationToken? cancellationToken;

  RunOptions({
    int? requestId,
    this.timeout,
    this.cancellationToken,
  }) : requestId = requestId ?? _generateRequestId();

  static int _nextId = 1;
  static int _generateRequestId() => _nextId++;
}

/// Token to cooperatively cancel an in-flight or queued inference run.
class CancellationToken {
  bool _isCancelled = false;
  final _completer = Completer<void>();

  /// Whether cancellation has been requested.
  bool get isCancelled => _isCancelled;

  /// Future that completes when cancellation is requested.
  Future<void> get onCancelled => _completer.future;

  /// Request cancellation of the associated operation.
  void cancel() {
    if (!_isCancelled) {
      _isCancelled = true;
      _completer.complete();
    }
  }
}

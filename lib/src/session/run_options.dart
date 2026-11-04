import 'dart:async';

/// Options for an individual inference execution run.
class RunOptions {
  /// Caller correlation ID. Duplicates are allowed; native task IDs control cancellation.
  final int requestId;

  /// Optional monotonic timeout including native submission/copying, queueing and run.
  final Duration? timeout;

  /// Optional cancellation token allowing cooperative cancellation of this run.
  final CancellationToken? cancellationToken;

  RunOptions({int? requestId, this.timeout, this.cancellationToken})
    : requestId = requestId ?? _generateRequestId();

  static int _nextId = 1;
  static int _generateRequestId() => _nextId++;
}

/// Token to cooperatively cancel an in-flight or queued inference run.
class CancellationToken {
  bool _isCancelled = false;
  final _listeners = <void Function()>{};
  Completer<void>? _completer;

  bool get isCancelled => _isCancelled;

  /// Compatibility notification. Inference uses removable listeners instead.
  Future<void> get onCancelled {
    final c = _completer ??= Completer<void>();
    if (_isCancelled && !c.isCompleted) c.complete();
    return c.future;
  }

  /// Returns an idempotent remover; completed tasks release all references.
  void Function() addListener(void Function() listener) {
    if (_isCancelled) {
      listener();
      return () {};
    }
    _listeners.add(listener);
    return () => _listeners.remove(listener);
  }

  void cancel() {
    if (_isCancelled) return;
    _isCancelled = true;
    if (_completer case final c? when !c.isCompleted) c.complete();
    final pending = _listeners.toList();
    _listeners.clear();
    for (final listener in pending) {
      listener();
    }
  }
}

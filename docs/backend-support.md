# Backend status (0.2.0)

| Target | Implemented route | Current acceptance |
|---|---|---|
| macOS arm64, macOS14+ | Native-assets build; pinned ORT CoreML/CPU | Stable CPU inference; CoreML experimental, GPU/ANE run observation not collected |
| iOS arm64 device | Explicit separate verified native bundle | Experimental; no device-specific artifact/consumer acceptance in this repair |
| iOS arm64 simulator | Explicit simulator bundle, never macOS dylib | Experimental; simulator cannot establish ANE use |
| Android arm64-v8a | Explicit ORT QNN/HTP bundle with actual ELF checks | Experimental; QNN/HTP SDK/build/package acceptance open |
| Linux x64/arm64 | Pinned CPU correctness artifacts | Configured; no desktop GPU claim |

CoreML accepts ALL, CPUAndGPU, CPUOnly, CPUAndNeuralEngine. These specify allowed physical units; none establishes which unit ran. QNN HTP selection is independent of Apple capabilities. Artifact EP enumeration describes compiled capabilities only. WebGPU may appear in the official artifact's actual enumeration; the bridge exposes no new WebGPU route.

No target is promoted to stable accelerated support by skipping device tests. See device matrix, generated artifact manifest, release gate, and remediation evidence. Earlier v0.1 "all platforms ready" and Hardware Observed statements are superseded; historical entries remain in Git and the 0.1 CHANGELOG is annotated.

Stable 0.2.0 acceptance is scoped to CPU inference on macOS arm64. The release checks and host consumer evidence are recorded in [stable-0.2.0.md](reviews/stable-0.2.0.md); other targets retain the limitations above.

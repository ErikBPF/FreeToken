# Contract only: unautomated until scenarios are bound to the focused FreeToken
# tests and guarded reference-machine recipes named in the validation plan.
@unautomated
Feature: Improve Qwen3.6 ROCm decode from a reproducible upstream-ready source

  Background:
    Given the test machine has 128 GB RAM, a Ryzen 9 5950X CPU, and an RX 9070 XT GPU using ROCm HIP
    And the exact Unsloth Qwen3.6-35B-A3B Q4_K_M artifact is verified by size and SHA-256
    And model residency is fixed at 0.50 with 4096 reserved KV tokens
    And context is 8192 tokens
    And the benchmark has a 2010-token prompt and exactly 1024 generated tokens
    And every disruptive experiment must restore the production llama server

  Scenario: Reconstruct the complete local oracle from immutable inputs
    Given the current local implementation and regression tests are preserved by one source artifact identity
    When an empty checkout is built on the test machine
    Then no dirty or untracked source is inherited from an older build
    And the build manifest identifies the source artifact, dependency freeze, runtime image, architecture, and every native library
    And preflight rejects mutation of any bound input

  Scenario: Select an official foundation without losing unique local behavior
    Given official ROCm and Qwen GGUF pull-request heads are pinned immutably
    When local changes are compared with each candidate source line
    Then every local runtime hunk and regression test is classified as adopt, adapt, retain, or drop
    And the selected foundation is the smallest clean composition that preserves required behavior
    And no unreviewed community fork executes on the test machine

  Scenario Outline: Reject an incomparable or unqualified performance result
    Given a candidate report has <invalid evidence>
    When the candidate is evaluated for retention
    Then it is rejected
    And it does not advance the performance target

    Examples:
      | invalid evidence                         |
      | a different model identity               |
      | an unqualified source identity           |
      | a different model residency              |
      | a different KV reservation               |
      | a different context                      |
      | a different prompt token count           |
      | a different completion token count       |
      | fewer or more than three samples         |
      | an invalid median                        |
      | a missing source or binary artifact hash |
      | empty or cache-corrupt functional output |

  Scenario: Preserve Qwen tokenizer and cache-state behavior
    When the same deterministic functional request runs from fresh and radix-cache states
    Then required special tokens remain atomic
    And sampling uses the final prompt position
    And each response is nonempty and structurally valid
    And any allowed output difference is recorded before throughput is graded

  Scenario: Preserve mixed GGUF and offload numerical behavior
    When a candidate changes loading, dispatch, offload, or the decode hot path
    Then unsupported quant dispatch fails closed
    And Q4_K gate and up output matches the independent path
    And Q5_K and Q6_K down output matches the independent path
    And Q8_0 output and reused Q8_1 activation match the independent path
    And mapped host-bank gathers are byte-exact
    And graph capture and replay remain correct with changed inputs
    And every output allocation is proven fully overwritten

  Scenario: Establish comparable llama.cpp and FreeToken endpoint baselines
    Given both backends use the same verified model and shared benchmark client
    When each backend completes one guarded three-sample run
    Then both reports pass their strict schemas and artifact checks
    And prompt, completion, context, and model identities match
    And production is restored healthy after each run
    And historical FreeToken performance is not assigned to a different source

  Scenario: Attribute the qualified decode path before selecting an optimization
    Given the converged source passes behavior and endpoint qualification
    When the decode workload is profiled
    Then the report fingerprints source, binaries, model, runtime, GPU, and fixed controls
    And it ranks expert copies, Q4_K, Q5_K, Q6_K, Q8_0, routing, graph, and launch overhead
    And it records profiler overhead and unclassified time

  Scenario: Retain only a measured improvement
    Given a candidate passes numerical, pointer, graph, and functional checks
    When the fixed benchmark runs three times
    Then a regression is reverted
    And a retained result records raw samples, median, and complete provenance
    And retained source remains recoverable independently of the benchmark host

  Scenario: Raise the single-stream target from the reproducible local baseline
    Given the active baseline is the latest reproducible qualified three-sample result
    When a qualified fixed benchmark reaches the active target
    Then the next target is exactly 10 median decode tokens per second higher
    And residency and workload controls remain unchanged

  Scenario: Overlap prefill when exceptional Q6 down banks are present
    Given the primary Q4 expert cache has two prefill buffers
    And each exceptional Q6 down bank remains a synchronous direct expert mapping
    When a Q6 layer runs prefill
    Then the primary layer is prefetched and released through the existing overlap choreography
    And the matching Q6 down layer is materialized before its fused expert operation
    And exact functional output, time to first token, decode throughput, and live VRAM are recorded

  Scenario Outline: Measure native concurrent decode without claiming a single-stream gain
    Given no additional expert weights are resident
    When <requests> requests decode concurrently through a graph that supports that batch size
    Then every request produces exactly 1024 tokens with valid deterministic output
    And aggregate decode throughput and per-request latency are reported separately
    And the report is compared with serialized requests on the same server

    Examples:
      | requests |
      | 2        |
      | 4        |

  Scenario: Select one launch or fusion candidate from the retained-path profile
    Given a bounded profile attributes the retained shared-overlap path
    When one dispatch family has measured single-stream headroom
    Then one isolated candidate changes that family only
    And its microbenchmark and exact endpoint result are recorded
    And a numerical regression, residency increase, or lower endpoint median reverts the candidate

  Scenario: Recover production after every disruptive experiment
    Given the production llama server is healthy before exclusive GPU access
    When a benchmark, profile, interruption, or validation step fails
    Then the FreeToken experiment is absent
    And the pinned production server reports healthy
    And failed recovery fails the experiment command

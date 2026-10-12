"""On-policy self-distillation and the safety tax: a scale-ladder study.

Modules:
    config    load and validate YAML run configs
    data      benign prompt pools, safety data, evaluation suites
    models    checkpoint loading, LoRA wiring, teacher/student pairing
    distill   the on-policy self-distillation training loop
    evaluate  capability and safety scoring, plus the NTR metric
    probes    refusal-direction geometry and output-entropy diagnostics
"""

__version__ = "0.1.0"

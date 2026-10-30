from llmverify.report import fidelity


def test_fidelity_scale_endpoints():
    assert fidelity(0.01, 0.01, 0.2) == 100.0
    assert fidelity(0.2, 0.01, 0.2) == 0.0
    assert fidelity(0.005, 0.01, 0.2) == 100.0  # closer than the honest reference: clipped
    assert fidelity(1.0, 0.01, 0.2) == 0.0
    mid = fidelity((0.01 * 0.2) ** 0.5, 0.01, 0.2)
    assert abs(mid - 50.0) < 0.2  # geometric midpoint scores 50
    assert fidelity(None, 0.01, 0.2) is None and fidelity(0.1, None, 0.2) is None


def test_prompt_generator_is_seeded_and_unique():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("gen", Path(__file__).resolve().parents[1] / "prompts" / "generate.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    a, b = gen.generate("s1", 40), gen.generate("s1", 40)
    assert a == b and len(set(a)) == 40 and all("{" not in p for p in a)
    assert gen.generate("s2", 40) != a

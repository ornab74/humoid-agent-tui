from humoid_tui.model_profiles import resolve_profile
from humoid_tui.prompts import build_system_prompt
from humoid_tui.config import Settings

def test_gpt56_profile():
    p=resolve_profile('openai','gpt-5.6-sol')
    assert p.key=='gpt56' and p.supports_programmatic_tools
    assert 'programmatic' in build_system_prompt(p).lower()

def test_gemma4_profile():
    p=resolve_profile('llamacpp','gemma-4-9b-it')
    assert p.key=='gemma4' and p.protocol=='gemma4'
    assert 'Gemma 4' in build_system_prompt(p)

def test_glm_profile():
    assert resolve_profile('digitalocean','glm-5.2').key=='glm52'

def test_muse_profile():
    assert resolve_profile('meta','muse-spark-1.1').key=='muse'

def test_model_specific_context_windows_are_independent():
    settings = Settings(
        digitalocean_model="glm-5.2",
        llamacpp_model="gemma-4-E2B_q4_0-it",
        humoid_gemma_context_limit=10000,
        humoid_glm_context_limit=20000,
    )
    assert settings.context_limit("llamacpp") == 10000
    assert settings.context_limit("digitalocean") == 20000

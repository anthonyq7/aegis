"""These test cover basic sanity tests: module imoports, return types, python version"""

import pytest


#test that preprocess module exists
def test_import_preprocess():
    try:
        import data.preprocess
        assert hasattr(data.preprocess, 'preprocess')
    except ModuleNotFoundError:
        pytest.skip("Required dependencies not installed")

#test that ataset module exists
def test_import_dataset():
    try:
        import model.dataset
        assert hasattr(model.dataset, 'CodeDataset')
    except ModuleNotFoundError:
        pytest.skip("Required dependencies not installed")

#test that prompts can be imported
def test_import_prompts():
    from data.prompts import build_message, get_questions
    assert callable(get_questions)
    assert callable(build_message)

#test that checks prompt functions return expected types
def test_prompt_functions_return_expected_types():
    from data.prompts import SYSTEM_PROMPT, build_message

    #SYSTEM_PROMPT should be a string
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0

    #build message should return a list
    message = build_message("question", "def func():")
    assert isinstance(message, list)
    assert len(message) == 2

#test python version
def test_environment():
    import sys
    assert sys.version_info >= (3, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_fast_action_suffix_reuses_action_code_ids():
    prompt = "Hello, world!"
    state = np.linspace(-1, 1, 5, dtype=np.float32)
    actions = np.linspace(-1, 1, 6, dtype=np.float32).reshape(3, 2)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)

    (full_tokens, _, _, _), full_metadata = tokenizer.tokenize_with_metadata(prompt, state, actions)
    suffix_tokens, suffix_mask, suffix_metadata = tokenizer.tokenize_action_suffix(actions)
    action_marker_length = (
        full_metadata.postfix_length
        - full_metadata.fast_action_code_count
        - (suffix_metadata.unpadded_length - suffix_metadata.fast_action_code_count)
    )
    old_action_start = full_metadata.prefix_length + action_marker_length
    old_action_ids = full_tokens[old_action_start : old_action_start + full_metadata.fast_action_code_count]
    new_action_ids = suffix_tokens[: suffix_metadata.fast_action_code_count]

    np.testing.assert_array_equal(new_action_ids, old_action_ids)
    assert suffix_metadata.fast_action_code_count == full_metadata.fast_action_code_count
    assert suffix_mask.sum() == suffix_metadata.unpadded_length
    assert not suffix_mask[suffix_metadata.unpadded_length :].any()


def test_fast_action_suffix_fails_instead_of_truncating():
    tokenizer = _tokenizer.FASTTokenizer(max_len=1)
    actions = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="refusing to truncate KI targets"):
        tokenizer.tokenize_action_suffix(actions)

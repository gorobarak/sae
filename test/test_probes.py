import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sae.probes import create_datasets
import tempfile
import os


@pytest.fixture
def setup_model_and_tokenizer():
    """Fixture to load a small model and tokenizer for testing."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype="auto",
        device_map="auto"
    )
    return model, tokenizer, model_name


@pytest.fixture
def create_mock_dataset(setup_model_and_tokenizer):
    """Fixture to create a mock preprocessed dataset."""
    _, tokenizer, model_name = setup_model_and_tokenizer
    
    # Create sample conversations
    sample_data = {
        "input_ids": [
            tokenizer.encode("Hello, how are you?", add_special_tokens=True),
            tokenizer.encode("What is AI?", add_special_tokens=True),
            tokenizer.encode("Tell me a joke", add_special_tokens=True),
            tokenizer.encode("Explain quantum physics", add_special_tokens=True),
        ]
    }
    
    dataset = Dataset.from_dict(sample_data)
    
    # Save to temporary directory
    dataset_name = "test_dataset"
    L_max = 256
    temp_dir = tempfile.mkdtemp()
    save_path = f"{temp_dir}/data/preprocessed/{dataset_name}/{model_name}_L={L_max}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    
    return temp_dir, dataset_name, L_max, len(sample_data["input_ids"])


def test_create_datasets_output_structure(setup_model_and_tokenizer, create_mock_dataset, monkeypatch):
    """Test that create_datasets returns correctly structured outputs."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    temp_dir, dataset_name, L_max, num_samples = create_mock_dataset
    
    # Monkeypatch to use temp directory
    monkeypatch.chdir(temp_dir)
    
    pooling_strategies = ["last", "mean"]
    dataset_size = num_samples
    batch_size = 2
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        pooling_strategies=pooling_strategies,
        dataset_size=dataset_size,
        batch_size=batch_size,
        max_new_tokens=10,
        L_max=L_max
    )
    
    # Check X structure
    assert isinstance(X, dict), "X should be a dictionary"
    assert set(X.keys()) == set(pooling_strategies), f"X should have keys {pooling_strategies}"
    
    for strategy in pooling_strategies:
        assert isinstance(X[strategy], dict), f"X[{strategy}] should be a dictionary"
        assert len(X[strategy]) == model.config.num_hidden_layers, \
            f"X[{strategy}] should have {model.config.num_hidden_layers} layers"
    
    # Check Y structure
    assert isinstance(Y, dict), "Y should be a dictionary"
    assert "response_token_ids" in Y, "Y should contain 'response_token_ids'"
    assert "lengths" in Y, "Y should contain 'lengths'"


def test_create_datasets_tensor_shapes(setup_model_and_tokenizer, create_mock_dataset, monkeypatch):
    """Test that tensors have correct shapes."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    temp_dir, dataset_name, L_max, num_samples = create_mock_dataset
    
    monkeypatch.chdir(temp_dir)
    
    dataset_size = num_samples
    batch_size = 2
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        pooling_strategies=["last"],
        dataset_size=dataset_size,
        batch_size=batch_size,
        max_new_tokens=10,
        L_max=L_max
    )
    
    # Check X tensor shapes
    for layer_idx in range(model.config.num_hidden_layers):
        acts = X["last"][layer_idx]
        assert acts.shape[0] == dataset_size, \
            f"Layer {layer_idx}: First dimension should be {dataset_size}"
        assert acts.shape[1] == model.config.hidden_size, \
            f"Layer {layer_idx}: Second dimension should be {model.config.hidden_size}"
        assert len(acts.shape) == 2, \
            f"Layer {layer_idx}: Activations should be 2D tensor"
    
    # Check Y tensor shapes
    assert Y["lengths"].shape[0] == dataset_size, \
        "Lengths tensor should have dataset_size elements"
    assert Y["response_token_ids"].shape[0] == dataset_size, \
        "Response token IDs tensor should have dataset_size elements"


def test_create_datasets_response_lengths(setup_model_and_tokenizer, create_mock_dataset, monkeypatch):
    """Test that response lengths are computed correctly."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    temp_dir, dataset_name, L_max, num_samples = create_mock_dataset
    
    monkeypatch.chdir(temp_dir)
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_size=num_samples,
        batch_size=2,
        max_new_tokens=10,
        L_max=L_max
    )
    
    lengths = Y["lengths"]
    
    # Lengths should be positive or -1 (for truncated)
    assert torch.all((lengths > 0) | (lengths == -1)), \
        "Lengths should be positive or -1"
    
    # Check consistency with response token IDs
    for i, length in enumerate(lengths):
        if length != -1:
            response = Y["response_token_ids"][i]
            # Length should not exceed response length
            assert length <= response.shape[0], \
                f"Computed length {length} exceeds response length {response.shape[0]}"


def test_create_datasets_multiple_pooling_strategies(setup_model_and_tokenizer, create_mock_dataset, monkeypatch):
    """Test that multiple pooling strategies work correctly."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    temp_dir, dataset_name, L_max, num_samples = create_mock_dataset
    
    monkeypatch.chdir(temp_dir)
    
    pooling_strategies = ["last", "mean", "max"]
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        pooling_strategies=pooling_strategies,
        dataset_size=num_samples,
        batch_size=2,
        max_new_tokens=10,
        L_max=L_max
    )
    
    # Check all pooling strategies are present
    for strategy in pooling_strategies:
        assert strategy in X, f"Pooling strategy '{strategy}' not found in X"
        
        # All strategies should have same number of layers
        assert len(X[strategy]) == model.config.num_hidden_layers, \
            f"Strategy '{strategy}' should have {model.config.num_hidden_layers} layers"
        
        # All strategies should have same shape
        for layer_idx in range(model.config.num_hidden_layers):
            assert X[strategy][layer_idx].shape == (num_samples, model.config.hidden_size), \
                f"Strategy '{strategy}', layer {layer_idx} has incorrect shape"


def test_create_datasets_device_handling(setup_model_and_tokenizer, create_mock_dataset, monkeypatch):
    """Test that outputs are on CPU regardless of model device."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    temp_dir, dataset_name, L_max, num_samples = create_mock_dataset
    
    monkeypatch.chdir(temp_dir)
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_size=num_samples,
        batch_size=2,
        max_new_tokens=10,
        L_max=L_max
    )
    
    # Check all tensors are on CPU
    for layer_idx in range(model.config.num_hidden_layers):
        assert X["last"][layer_idx].device == torch.device("cpu"), \
            f"Layer {layer_idx} activations should be on CPU"
    
    assert Y["lengths"].device == torch.device("cpu"), \
        "Lengths tensor should be on CPU"
    assert Y["response_token_ids"].device == torch.device("cpu"), \
        "Response token IDs should be on CPU"


def test_create_datasets_long_and_short_answers(setup_model_and_tokenizer, monkeypatch):
    """Test that create_datasets handles mixed short and long responses in the same batch."""
    model, tokenizer, model_name = setup_model_and_tokenizer
    
    # Create dataset with prompts that will likely generate different length responses
    # Mix short and long prompts to test batch handling
    sample_data = {
        "input_ids": [
            tokenizer.encode("Say yes.", add_special_tokens=True),  # Short
            tokenizer.encode("Write a detailed explanation of machine learning.", add_special_tokens=True),  # Long
            tokenizer.encode("What is 2+2?", add_special_tokens=True),  # Short
            tokenizer.encode("Describe the history of computer science in detail.", add_special_tokens=True),  # Long
            tokenizer.encode("No.", add_special_tokens=True),  # Very short
            tokenizer.encode("Explain quantum mechanics, relativity, and string theory.", add_special_tokens=True),  # Long
        ]
    }
    
    dataset = Dataset.from_dict(sample_data)
    
    # Save to temporary directory
    dataset_name = "test_mixed_lengths"
    L_max = 256
    temp_dir = tempfile.mkdtemp()
    save_path = f"{temp_dir}/data/preprocessed/{dataset_name}/{model_name}_L={L_max}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    
    monkeypatch.chdir(temp_dir)
    
    # Process all samples in batches that will contain both short and long responses
    num_samples = len(sample_data["input_ids"])
    batch_size = 3  # Each batch will have mixed lengths
    
    X, Y = create_datasets(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_size=num_samples,
        batch_size=batch_size,
        max_new_tokens=50,
        L_max=L_max
    )
    
    # Check that we got results for all samples
    assert Y["lengths"].shape[0] == num_samples, \
        f"Should have {num_samples} responses, got {Y['lengths'].shape[0]}"
    assert Y["response_token_ids"].shape[0] == num_samples, \
        f"Should have {num_samples} response token IDs"
    
    # Check that lengths vary (we should have both short and long responses)
    lengths = Y["lengths"]
    valid_lengths = lengths[lengths > 0]  # Exclude truncated responses
    
    if len(valid_lengths) > 1:
        min_length = valid_lengths.min().item()
        max_length = valid_lengths.max().item()
        
        # We should have variation in response lengths
        assert max_length > min_length, \
            f"Expected variation in response lengths, but got min={min_length}, max={max_length}"
        
        # Check that the range is reasonable (not all identical)
        assert max_length >= min_length * 2, \
            "Expected significant variation between short and long responses"
    
    # Verify all activations have correct shapes
    for layer_idx in range(model.config.num_hidden_layers):
        acts = X["last"][layer_idx]
        assert acts.shape[0] == num_samples, \
            f"Layer {layer_idx}: Expected {num_samples} samples, got {acts.shape[0]}"
        assert acts.shape[1] == model.config.hidden_size, \
            f"Layer {layer_idx}: Expected hidden size {model.config.hidden_size}, got {acts.shape[1]}"
    
    # Check that each response token IDs tensor has the correct shape
    for i in range(num_samples):
        response = Y["response_token_ids"][i]
        length = Y["lengths"][i].item()
        
        if length > 0:  # Not truncated
            # Response should have length <= max_new_tokens
            assert response.shape[0] <= 50, \
                f"Response {i} length {response.shape[0]} exceeds max_new_tokens=50"
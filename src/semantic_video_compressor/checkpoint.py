import torch
from pathlib import Path
from neural_atlases.implicit_neural_networks import IMLP
import quanto

def normalize_checkpoint(run_dir):
    """
    Create compressed checkpoint by removing large state dicts.
    
    Args:
        run_dir (Path): Experiment directory from training results
    """    
    checkpoint_path = Path(run_dir / "checkpoint")
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Remove large state dictionaries to save space
    checkpoint.pop("F_atlas_state_dict", None)
    checkpoint.pop("optimizer_all_state_dict", None)
    
    torch.save(checkpoint, checkpoint_path)
    
def quantize_checkpoint(config, device, input_checkpoint_path=None, output_checkpoint_path=None):
    # M_\alpha's hyper parameters:
    positional_encoding_num_alpha = config["positional_encoding_num_alpha"]
    number_of_channels_alpha = config["number_of_channels_alpha"]
    number_of_layers_alpha = config["number_of_layers_alpha"]

    # M_f's hyper parameters
    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    # M_b's hyper parameters
    use_positional_encoding_mapping2 = config["use_positional_encoding_mapping2"]
    number_of_positional_encoding_mapping2 = config["number_of_positional_encoding_mapping2"]
    number_of_layers_mapping2 = config["number_of_layers_mapping2"]
    number_of_channels_mapping2 = config["number_of_channels_mapping2"]

    model_F_mapping1 = IMLP(
            input_dim=3,
            output_dim=2,
            hidden_dim=number_of_channels_mapping1,
            use_positional=use_positional_encoding_mapping1,
            positional_dim=number_of_positional_encoding_mapping1,
            num_layers=number_of_layers_mapping1,
            skip_layers=[]).to(device)

    model_F_mapping2 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping2,
        use_positional=use_positional_encoding_mapping2,
        positional_dim=number_of_positional_encoding_mapping2,
        num_layers=number_of_layers_mapping2,
        skip_layers=[]).to(device)

    model_alpha = IMLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=number_of_channels_alpha,
        use_positional=True,
        positional_dim=positional_encoding_num_alpha,
        num_layers=number_of_layers_alpha,
        skip_layers=[]).to(device)

    checkpoint = torch.load(input_checkpoint_path, map_location="cpu")

    model_F_mapping1.load_state_dict(checkpoint['model_F_mapping1_state_dict'])
    model_F_mapping2.load_state_dict(checkpoint['model_F_mapping2_state_dict'])
    model_alpha.load_state_dict(checkpoint['model_F_alpha_state_dict'])

    model_F_mapping1.quantize()
    model_F_mapping2.quantize()
    model_alpha.quantize()

    # print(checkpoint.keys())

    # Save quantized models
    torch.save({
        "model_F_mapping1_state_dict": model_F_mapping1.state_dict(),
        "model_F_mapping2_state_dict": model_F_mapping2.state_dict(),
        "model_F_alpha_state_dict"   : model_alpha.state_dict(),
    }, output_checkpoint_path)

    print(f"Quantized checkpoint saved to: {input_checkpoint_path} -> {output_checkpoint_path}")
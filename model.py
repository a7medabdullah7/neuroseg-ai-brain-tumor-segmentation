
from pathlib import Path

import torch
import segmentation_models_pytorch as smp


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "brain_tumor_pretrained_unet_resnet34.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.6


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("Model metadata:")
    for key in [
        "model_name",
        "architecture",
        "encoder_name",
        "encoder_weights",
        "in_channels",
        "classes",
        "activation",
        "threshold",
        "best_epoch",
        "test_positive_dice",
        "test_positive_iou",
        "test_empty_mask_fpr",
    ]:
        print(f"{key}: {checkpoint.get(key)}")

    model = smp.Unet(
        encoder_name=checkpoint["encoder_name"],
        encoder_weights=None,
        in_channels=checkpoint["in_channels"],
        classes=checkpoint["classes"],
        activation=checkpoint["activation"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    threshold = float(checkpoint["threshold"])

    return model, threshold


if __name__ == "__main__":
    model, threshold = load_model()

    print("\nModel loaded successfully!")
    print("Device:", DEVICE)
    print("Threshold:", threshold)

    dummy_input = torch.randn(1, 1, 256, 256).to(DEVICE)

    with torch.inference_mode():
        output = model(dummy_input)

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Output is finite:", torch.isfinite(output).all().item())

import torch
import logging


class DeviceManager:
    """Class responsible for automated and safe hardware device management."""

    @staticmethod
    def get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            device_type = "cuda"
            # Enable cuDNN benchmark for optimized performance on fixed-size inputs
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        logging.info(f"Selected hardware accelerator: {device_type.upper()}")
        return torch.device(device_type)

    @staticmethod
    def cleanup_memory(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
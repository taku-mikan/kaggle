import segmentation_models_pytorch as smp
import torch

from models.vision_transformer import SwinUnet

def build_model(backbone, num_classes, device, img_size, model="Unet", config=None):
    if model == "Unet":
        model = smp.Unet(
            encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,        # model output channels (number of classes in your dataset)
            activation=None,
        )
    elif model == "SwinUnet":
        model = SwinUnet(config, img_size, num_classes)

    model.to(device)
    return model

def load_model(path, backbone, num_classes, device, img_size, model, config):
    model = build_model(backbone, num_classes, device, img_size, model, config)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

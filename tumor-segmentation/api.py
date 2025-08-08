from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
import numpy as np
from utils import validate_segmentation, encode_request, decode_request
from pydantic import BaseModel
import torch
from main import (
    TumorModel,
    reverse_image_transform,
    pad_image_to_size,
    DESIRED_WIDTH,
    DESIRED_HEIGHT,
)
import time


class PredictRequestDto(BaseModel):
    img: str


class PredictResponseDto(BaseModel):
    img: str


# 2 = 0.75 (Unet++ + EfficientNet B5)
# 4 = 0.72 (Deeplabv3+ + Resnet50)

app = FastAPI()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load both models for ensemble
model_2 = TumorModel.load_from_checkpoint("5.ckpt")
model_2.to(device)
model_2.eval()

# model_4 = TumorModel.load_from_checkpoint("4.ckpt")
# model_4.to(device)
# model_4.eval()


@app.post("/predict", response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):
    start = time.time()
    original_img: np.ndarray = decode_request(request)
    original_width = original_img.shape[1]
    original_height = original_img.shape[0]

    img = original_img[:, :, 0]
    img = pad_image_to_size(img, DESIRED_HEIGHT, DESIRED_WIDTH)
    img = torch.tensor(img, dtype=torch.float32) / 255
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.to(device)

    # Get logits from both models
    with torch.no_grad():
        logits_mask = model_2.forward(img)
        # logits_mask_4 = model_4.forward(img)

        # # Average the logits from both models
        # logits_mask = (logits_mask_2 + logits_mask_4) / 2.0

    prob_mask = logits_mask.sigmoid()

    pred_mask = (prob_mask > 0.5).float()
    pred_mask = pred_mask.squeeze(0)
    pred_mask = reverse_image_transform(pred_mask, original_width, original_height)
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask = pred_mask.cpu().numpy()
    # Validate segmentation format
    validate_segmentation(original_img, pred_mask)
    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(pred_mask)
    # Return the encoded segmentation to the validation/evalution service
    response = PredictResponseDto(img=encoded_segmentation)
    end = time.time()
    print(f"Time: {end - start:2f}")
    return response


# Set the port for your server
port = 8000

# Setup ngrok tunnel
ngrok_tunnel = ngrok.connect(port)
print()
print()
print(ngrok_tunnel.public_url)
print()
print()

# Run the server
uvicorn.run(app, port=port)

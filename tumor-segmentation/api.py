from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
import numpy as np
from utils import validate_segmentation, encode_request, decode_request
from pydantic import BaseModel
import torch
from main import TumorModel, reverse_image_transform, image_transform
import time


class PredictRequestDto(BaseModel):
    img: str


class PredictResponseDto(BaseModel):
    img: str


# 1 = 0.73
# 2 = 0.727
# 3 = 0.720

app = FastAPI()
ckpt_path = "final_model_default-3.ckpt"
model = TumorModel.load_from_checkpoint(ckpt_path)
model.eval()


@app.post("/predict", response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):
    start = time.time()
    original_img: np.ndarray = decode_request(request)
    original_width = original_img.shape[1]
    original_height = original_img.shape[0]

    img = original_img[:, :, 0]
    img = image_transform(img)
    img = torch.tensor(img, dtype=torch.float32) / 255
    img = img.unsqueeze(0).unsqueeze(0)
    logits_mask = model.forward(img)

    prob_mask = logits_mask.sigmoid()

    pred_mask = (prob_mask > 0.5).float()
    pred_mask = pred_mask.squeeze(0)
    pred_mask = reverse_image_transform(pred_mask, original_width, original_height)
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask = pred_mask.numpy()
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

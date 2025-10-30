# api.py (FastAPI)
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import io
import torch
from torchvision import models, transforms
import numpy as np

app = FastAPI()
model = None
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.on_event("startup")
def load():
    global model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu")))
    model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1).numpy()[0]
    return {"classes": class_names, "probs": probs.tolist(), "predicted": class_names[int(probs.argmax())]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

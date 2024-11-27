import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

MODEL_WEIGHT_PATH="best.pt"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*" ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
  "/api/classify",  
  responses = { 200: { "content": {"image/png": {}} }}
)
async def read_item(file: UploadFile):
  if file.content_type != "image/jpeg" and file.content_type != "image/png":
    raise ValueError("Invalid file type")
  
  fine_tuned_model4 = YOLO(MODEL_WEIGHT_PATH)
  results = fine_tuned_model4(Image.open(BytesIO(await file.read())))
  
  image_data = BytesIO()

  for result in results:
    plt.imshow(result.plot())
    plt.axis("off")
    plt.savefig(image_data, format='PNG', bbox_inches='tight')

  image_data.seek(0)
  return Response(content=image_data.read(), media_type="image/png")

app.mount("/", StaticFiles(directory="webapp/dist", html=True), name="static")
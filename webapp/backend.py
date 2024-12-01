import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, Response, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from os import listdir
from os.path import isfile, join
from typing_extensions import Annotated

base = "./weights"
models = [f for f in listdir(base) if not isfile(join(base, f))]

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
async def read_item(file: UploadFile, model: Annotated[str, Form()]):
  if file.content_type != "image/jpeg" and file.content_type != "image/png":
    raise ValueError("Invalid file type")
  if model not in models:
    raise ValueError("Invalid model")
  
  fine_tuned_model4 = YOLO(join(base, model, "best.pt"))
  results = fine_tuned_model4(Image.open(BytesIO(await file.read())))
  
  image_data = BytesIO()

  for result in results:
    plt.imshow(result.plot())
    plt.axis("off")
    plt.savefig(image_data, format='PNG', bbox_inches='tight')

  image_data.seek(0)
  return Response(content=image_data.read(), media_type="image/png")

app.mount("/", StaticFiles(directory="dist", html=True), name="static")
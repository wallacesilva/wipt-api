from fastapi import FastAPI, File, UploadFile
from rembg import remove

app = FastAPI()


@app.post("/files/")
async def create_file(image: bytes = File()):
    return {"file_size": len(image)}


@app.post("/uploadfile/")
async def create_upload_file(image: UploadFile):
    return {"filename": image.filename}


@app.post("/removebg/")
async def removebg_from_image(image: UploadFile):
    if not image:
        return {"message": "No file sent"}
    else:
        if image.content_type not in ['image/png', 'image/jpg', 'image/jpeg']:
            return {"error": "File not allowed. Try image JPG or PNG"}
        contents = await image.read()
        output_path = f"data/{image.filename}_bgrem.png"
        with open(output_path, 'wb') as o:
            output = remove(contents)
            o.write(output)
        return {"filename": image.filename}


@app.post("/anime/")
async def anime_from_image(image: UploadFile):
    if not image:
        return {"message": "No file sent"}
    else:
        if image.content_type not in ['image/png', 'image/jpg', 'image/jpeg']:
            return {"error": "File not allowed. Try image JPG or PNG"}
        contents = await image.read()
        output_path = "data/bg_removed.png"
        with open(output_path, 'wb') as o:
            output = remove(contents)
            o.write(output)
        return {"filename": image.filename}
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime import InferenceSession
import csv
import aiohttp
import asyncio

class Tagger:
    models_dir = "models"
    defaults = {
        "model": "wd-v1-4-vit-tagger-v2",
        "threshold": 0.35,
        "character_threshold": 0.85,
        "replace_underscore": False,
        "trailing_comma": False,
        "exclude_tags": ""
    }
    all_models = (
        "wd-v1-4-moat-tagger-v2", 
        "wd-v1-4-convnext-tagger-v2", "wd-v1-4-convnext-tagger",
        "wd-v1-4-convnextv2-tagger-v2", "wd-v1-4-vit-tagger-v2"
    )
    loaded_models = {}

    def __init__(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    async def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        installed = list(self.get_installed_models())
        name = os.path.join(self.models_dir, model_name + ".onnx")
        if not any(model_name + ".onnx" in s for s in installed):
            await self.download_model(model_name)

        session = InferenceSession(name, providers=ort.get_available_providers())
        self.loaded_models[model_name] = session
        return session

    @staticmethod
    def get_ext_dir(subpath=None, mkdir=False):
        dir = os.path.dirname(__file__) if '__file__' in locals() else ''
        if subpath is not None:
            dir = os.path.join(dir, subpath)
        dir = os.path.abspath(dir)
        if mkdir and not os.path.exists(dir):
            os.makedirs(dir)
        return dir


    def get_installed_models(self):
        return filter(lambda x: x.endswith(".onnx"), os.listdir(self.models_dir))


    async def download_to_file(self, url, destination, is_ext_subpath=True, session=None):
        close_session = False
        if session is None:
            close_session = True
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            session = aiohttp.ClientSession(loop=loop)
        if is_ext_subpath:
            destination = self.get_ext_dir(destination)
        try:
            async with session.get(url) as response:
                total = int(response.headers.get('content-length', 0)) or None
                n = 0

                with open(destination, mode='wb') as f:
                    perc = 0
                    perc_last = ""
                    
                    async for chunk in response.content.iter_chunked(2048):
                        f.write(chunk)
                        n += len(chunk)
                        perc = round(n / total, 2)
                        parc_str = f"{perc*100:.0f}%"
                        if parc_str != perc_last:
                            perc_last = parc_str
                            print(f"Downloaded {n} of {total} bytes ({parc_str})")

        finally:
            if close_session and session is not None:
                await session.close()

    def tag_sync(self, image, model_name=None, threshold=None, character_threshold=None, exclude_tags="", replace_underscore=True, trailing_comma=False):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.tag(image, model_name, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma))
        return result

    async def tag(self, image, model_name=None, threshold=None, character_threshold=None, exclude_tags="", replace_underscore=True, trailing_comma=False):
        if model_name is None:
            model_name = self.defaults["model"]
        
        if model_name not in self.all_models:
            raise ValueError(f"Model {model_name} not found")
        
        if threshold is None:
            threshold = self.defaults["threshold"]

        if character_threshold is None:
            character_threshold = self.defaults["character_threshold"]
        
        model = await self.load_model(model_name)
        input = model.get_inputs()[0]
        height = input.shape[1]

        # Reduce to max size and pad with white
        ratio = float(height)/max(image.size)
        new_size = tuple([int(x*ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (height, height), (255, 255, 255))
        square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

        image = np.array(square).astype(np.float32)
        image = image[:, :, ::-1]  # RGB -> BGR
        image = np.expand_dims(image, 0)

        tags = []
        general_index = None
        character_index = None
        with open(os.path.join(self.models_dir, model_name + ".csv")) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                if replace_underscore:
                    tags.append(row[1].replace("_", " "))
                else:
                    tags.append(row[1])

        label_name = model.get_outputs()[0].name
        probs = model.run([label_name], {input.name: image})[0]

        result = list(zip(tags, probs[0]))

        general = [item for item in result[general_index:character_index] if item[1] > threshold]
        character = [item for item in result[character_index:] if item[1] > character_threshold]

        all = character + general
        remove = [s.strip() for s in exclude_tags.lower().split(",")]
        all = [tag for tag in all if tag[0] not in remove]

        res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

        return res


    async def download_model(self, model):
        url = f"https://huggingface.co/SmilingWolf/{model}/resolve/main/"
        async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
            await self.download_to_file(
                f"{url}model.onnx", os.path.join("models",f"{model}.onnx"), session=session)
            await self.download_to_file(
                f"{url}selected_tags.csv", os.path.join("models",f"{model}.csv"), session=session)
            print(f"Downloaded {model}")

        return


async def main():
    tagger = Tagger()

    print(await tagger.tag(Image.open("test1.png")))
    print(await tagger.tag(Image.open("test2.png")))


if __name__ == "__main__":
    asyncio.run(main())


# Path: main.py

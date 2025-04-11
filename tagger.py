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
        "model": "wd-eva02-large-tagger-v3",
        "threshold": 0.35,
        "character_threshold": 0.85,
        "replace_underscore": False,
        "trailing_comma": False,
        "exclude_tags": ""
    }
    all_models = (
        "wd-v1-4-moat-tagger-v2", "wd-v1-4-vit-tagger",
        "wd-v1-4-convnext-tagger-v2", "wd-v1-4-convnext-tagger",
        "wd-v1-4-convnextv2-tagger-v2", "wd-v1-4-vit-tagger-v2",
        "wd-v1-4-swinv2-tagger-v2", "wd-vit-tagger-v3",
        "wd-swinv2-tagger-v3", "wd-convnext-tagger-v3",
        "wd-eva02-large-tagger-v3"
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
        print(f"Available Providers:{ort.get_available_providers()}")
        print(f"Selected Providers:{session.get_providers()}")
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
                with open(destination, mode='wb') as f:
                    async for chunk in response.content.iter_chunked(2048):
                        f.write(chunk)

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

    async def tag_batch(self, images: list[tuple[str, Image.Image]], model_name=None, threshold=None, character_threshold=None, exclude_tags="", replace_underscore=True, trailing_comma=False):
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

        # 画像たちを前処理してリスト化
        prepared = []
        filenames = []  # ファイル名リスト
        for filename, img in images:
            ratio = float(height) / max(img.size)
            new_size = tuple([int(x * ratio) for x in img.size])
            img = img.resize(new_size, Image.LANCZOS)
            square = Image.new("RGB", (height, height), (255, 255, 255))
            square.paste(img, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
            img = np.array(square).astype(np.float32)
            img = img[:, :, ::-1]  # RGB -> BGR
            prepared.append(img)
            filenames.append(filename)

        # まとめてバッチ化
        batch = np.stack(prepared, axis=0)

        # タグの読み込み
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

        # バッチ推論
        probs = model.run([label_name], {input.name: batch})[0]

        results = []
        for filename, prob in zip(filenames, probs):  # ファイル名と一緒に
            result = list(zip(tags, prob))
            general = [item for item in result[general_index:character_index] if item[1] > threshold]
            character = [item for item in result[character_index:] if item[1] > character_threshold]

            all_tags = character + general
            remove = [s.strip() for s in exclude_tags.lower().split(",")]
            all_tags = [tag for tag in all_tags if tag[0] not in remove]

            res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all_tags))
            results.append((filename, res))  # ファイル名とタグ結果のペアで返す

        return results


    async def download_model(self, model):
        url = f"https://huggingface.co/SmilingWolf/{model}/resolve/main/"
        print(f"Downloading {model} ( {url} )")
        async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
            await self.download_to_file(
                f"{url}model.onnx", os.path.join("models",f"{model}.onnx"), session=session)
            await self.download_to_file(
                f"{url}selected_tags.csv", os.path.join("models",f"{model}.csv"), session=session)
            print(f"Downloaded {model}")

        return

async def main():
    image_filenames = ["test1.png", "test2.png", "test3.jpg", "test4.jpg"]
    tagger = Tagger()

    import time
    start_time = time.time()
    for image_filename in image_filenames:
        print(await tagger.tag(Image.open(image_filename)))
    print(f"Single Image Time: {time.time() - start_time}")
    
    start_time = time.time()
    images = [(filename, Image.open(filename)) for filename in image_filenames]
    results = await tagger.tag_batch(images)

    for filename, tags in results:
        print(f"{filename}: {tags}")
    print(f"Batch Image Time: {time.time() - start_time}")


if __name__ == "__main__":
    asyncio.run(main())


# Path: main.py

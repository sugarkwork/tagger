import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自動で初期化
from PIL import Image
import csv
import os
import subprocess
import requests
import os
import multiprocessing as mp
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTTagger:
    def __init__(self, model="wd-eva02-large-tagger-v3", models_dir=None):
        if models_dir is not None:
            self.models_dir = models_dir
        # エンジン読み込み
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.download_model(model)
        engine_name = f"{model}.trt"
        tag_csv_name = f"{model}.csv"
        engine_path = os.path.join(self.models_dir, engine_name)
        tag_csv_path = os.path.join(self.models_dir, tag_csv_name)

        if not os.path.exists(engine_path):
            self.convert(f"{model}.onnx", engine_name)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # タグリスト読み込み
        self.tags, self.general_index, self.character_index = self.load_tags(tag_csv_path)

    def load_engine(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return runtime.deserialize_cuda_engine(engine_data)

    def load_tags(self, csv_path):
        tags = []
        general_index = None
        character_index = None
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])
        return tags, general_index, character_index

    def preprocess(self, img: Image.Image):
        h = self.input_shape[1]
        ratio = float(h) / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img = img.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (h, h), (255, 255, 255))
        square.paste(img, ((h - new_size[0]) // 2, (h - new_size[1]) // 2))
        img = np.array(square).astype(np.float32)
        img = img[:, :, ::-1]  # RGB → BGR
        return img

    def infer_batch(self, images: list[tuple[str, Image.Image]], threshold=0.35, character_threshold=0.85, batch_size=4):
        results = []
        if images is None:
            return results
        if isinstance(images, str):
            images = self.load_images([images])
        if isinstance(images, list):
            if len(images) == 0:
                return results
        if isinstance(images[0], str):
            images = self.load_images(images)
        if isinstance(images, Image.Image):
            images = [("image", images)]
        if isinstance(images[0], Image.Image):
            images = [(f"image{i}", img) for i,img in enumerate(images)]

        filenames_all = [filename for filename, _ in images]
        preprocessed_all = [self.preprocess(img) for _, img in images]

        # バッチサイズごとに分割して推論する
        for i in range(0, len(images), batch_size):
            filenames = filenames_all[i:i+batch_size]
            batch = np.stack(preprocessed_all[i:i+batch_size], axis=0)
            batch = np.ascontiguousarray(batch)  # メモリ連続化

            # 入力・出力用メモリ確保
            d_input = cuda.mem_alloc(batch.nbytes)
            d_output = cuda.mem_alloc(batch.shape[0] * self.output_shape[1] * 4)  # float32=4バイト

            bindings = [int(d_input), int(d_output)]

            cuda.memcpy_htod(d_input, batch)
            self.context.set_input_shape(self.input_name, batch.shape)  # 明示的にシェイプをセット
            self.context.execute_v2(bindings)

            output = np.empty((batch.shape[0], self.output_shape[1]), dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)

            # 1枚ずつ結果をまとめる
            for filename, probs in zip(filenames, output):
                tags_result = self.postprocess(probs, threshold, character_threshold)
                results.append((filename, tags_result))

        return results

    def postprocess(self, probs, threshold, character_threshold):
        result = list(zip(self.tags, probs))
        general = [item for item in result[self.general_index:self.character_index] if item[1] > threshold]
        character = [item for item in result[self.character_index:] if item[1] > character_threshold]
        all_tags = character + general
        res = ", ".join(item[0] for item in all_tags)
        return res
    
    def convert(self, onnx_path, trt_path):
        if os.path.exists(os.path.join(self.models_dir, trt_path)):
            return
        # convert
        cmd = f"trtexec --onnx={os.path.join(self.models_dir, onnx_path)} --saveEngine={os.path.join(self.models_dir, trt_path)} --minShapes=input:1x448x448x3 --optShapes=input:4x448x448x3 --maxShapes=input:8x448x448x3 --verbose"
        subprocess.run(cmd, shell=True)
    
    @staticmethod
    def load_images(image_paths: list[str]) -> list[tuple[str, Image.Image]]:
        return [(path, Image.open(path)) for path in image_paths]

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

    def download_model(self, model):
        installed = list(self.get_installed_models())
        if any(model + ".onnx" in s for s in installed):
            return

        url = f"https://huggingface.co/SmilingWolf/{model}/resolve/main/"
        print(f"Downloading {model} ( {url} )")
        
        self.download_to_file(
            f"{url}model.onnx", os.path.join(self.models_dir, f"{model}.onnx"))
        self.download_to_file(
            f"{url}selected_tags.csv", os.path.join(self.models_dir, f"{model}.csv"))
        print(f"Downloaded {model}")
        return

    def download_to_file(self, url, destination, is_ext_subpath=True):
        if is_ext_subpath:
            destination = self.get_ext_dir(destination)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # エラーがあれば例外を投げる
            
            with open(destination, mode='wb') as f:
                for chunk in response.iter_content(chunk_size=2048):
                    if chunk:  # 空のチャンクをスキップ
                        f.write(chunk)
        
        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}")
            raise


def main():
    # path setting
    # $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34\bin" + ";" + "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34\lib" + ";" + $env:PATH

    # convert
    # trtexec --onnx=models/wd-eva02-large-tagger-v3.onnx --saveEngine=models/wd-eva02-large-tagger-v3.trt --minShapes=input:1x448x448x3 --optShapes=input:4x448x448x3 --maxShapes=input:8x448x448x3 --verbose

    # trt model version check
    # trtexec --getPlanVersionOnly --loadEngine=models/wd-eva02-large-tagger-v3.trt
    
    print("TensorRT version:", trt.__version__)

    import time
    import glob
    import json

    image_filenames = glob.glob("test*.png") + glob.glob("test*.jpg")

    tagger = TensorRTTagger(model="wd-eva02-large-tagger-v3")

    print()
    start_time = time.time()
    for image in image_filenames:
        results = tagger.infer_batch(image)
        print(json.dumps(results, indent=4))
    print(f"elapsed time: {time.time() - start_time}")

    print()
    start_time = time.time()
    results = tagger.infer_batch(image_filenames)
    print(json.dumps(results, indent=4))
    print(f"elapsed time: {time.time() - start_time}")


if __name__ == "__main__":
    main()

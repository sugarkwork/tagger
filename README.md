# tagger
https://github.com/pythongosssss/ComfyUI-WD14-Tagger から、コードを削除し、標準的な Python からライブラリとして呼び出しやすくしただけのコードです。

画像を渡すと DeepDanbooru みたいなタグが取得できます。

# 使い方

同期型

    def main():
        tagger = Tagger()
      
        print(tagger.tag_sync(Image.open("test1.png")))
        print(tagger.tag_sync(Image.open("test2.png")))
      
      
    if __name__ == "__main__":
        main()

非同期型
    
    async def main():
        tagger = Tagger()
    
        print(await tagger.tag(Image.open("test1.png")))
        print(await tagger.tag(Image.open("test2.png")))
    
    
    if __name__ == "__main__":
        asyncio.run(main())

# 出力例

## 画像1
![1girl](test1.png)

### タグ
1girl, solo, long hair, breasts, looking at viewer, blush, open mouth, bangs, simple background, long sleeves, dress, ribbon, twintails, medium breasts, hair ribbon, upper body, white hair, sidelocks, frills, fang, puffy sleeves, pink eyes, grey background, apron, black dress, red ribbon, maid, maid headdress, juliet sleeves, white apron, maid apron, collared dress

## 画像2

![1girl](test2.png)

### タグ
1girl, solo, long hair, looking at viewer, smile, brown hair, brown eyes, closed mouth, collarbone, upper body, nude, outdoors, day, blurry, lips, depth of field, blurry background, realistic

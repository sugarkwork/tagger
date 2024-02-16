# tagger
https://github.com/pythongosssss/ComfyUI-WD14-Tagger から、コードを削除し、標準的な Python からライブラリとして呼び出しやすくしただけのコードです。

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

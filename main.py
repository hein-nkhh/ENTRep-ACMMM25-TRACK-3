from circle_cropper import Circle_Cropper

def main():
    cropper = Circle_Cropper()
    result = cropper.crop(r"E:\Project\ENTRep-ACMMM25-TRACK-3\Sample_data\Image\00ae1047-d0c0-46f4-96c8-798d01c74249.png")

    if result is not None:
        print("Ảnh đã crop xong!")
    else:
        print("Không tìm thấy hình tròn.")

if __name__ == "__main__":
    main()
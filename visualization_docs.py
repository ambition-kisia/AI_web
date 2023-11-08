import os
from PIL import Image

file_folder = "uploaded_file/"
image_folder = "static/images/"

def visualization():
    header_size = 65536
    
    for filename in os.listdir(file_folder):
        file_path = os.path.join(file_folder, filename)
        
        with open(file_path, 'rb') as file:
            header_data = file.read(header_size)
            
        num_pixels = header_size
        width = min(num_pixels, 256)
        height = (num_pixels + width - 1) // width
        
        grayscale_data = [int(byte) for byte in header_data]
        
        image = Image.new('L', (width, height))
        image.putdata(grayscale_data)
        
        image_filename = os.path.splitext(filename)[0] + '.png'
        image_path = os.path.join(image_folder, image_filename)
        image.save(image_path)
        
    print("이미지 변환 완료")
    
    return image_path
import os
import numpy as np
from PIL import Image

file_folder = "uploaded_file/"
image_folder = "static/images/"

def visualization():
    for filename in os.listdir(file_folder):
        if filename.endswith('.bytes'):
            array = []
            
            file_path = os.path.join(file_folder, filename)
            
            with open(file_path, 'rb') as file:
                lines = file.readlines()
            
            for line in lines:
                hex_values = line.split()[1:17]
                for hex_val in hex_values:
                    if hex_val != b'??':
                        array.append(int(hex_val, 16))
                        
            file_size = os.path.getsize(file_path)
            
            if file_size < 10240:
                width = 32
            elif 10240 <= file_size <= 10240 * 3:
                width = 64
            elif 10240 * 3 <= file_size <= 10240 * 6:
                width = 128
            elif 10240 * 6 <= file_size <= 10240 * 10:
                width = 256
            elif 10240 * 10 <= file_size <= 10240 * 20:
                width = 384
            elif 10240 * 20 <= file_size <= 10240 * 50:
                width = 512
            elif 10240 * 50 <= file_size <= 10240 * 100:
                width = 768
            else:
                width = 1024
            
            column_length = width
            row_length = column_length
            
            row_length = len(array) // column_length
            if len(array) % column_length != 0:
                row_length += 1
            
            two_dimensional_array = np.zeros((row_length, column_length), dtype=np.uint8)
            
            for i in range(len(array)):
                row_idx = i // column_length
                col_idx = i % column_length
                two_dimensional_array[row_idx][col_idx] = array[i]
            
            two_dimensional_array = np.uint8(two_dimensional_array)
            
            image = Image.fromarray(two_dimensional_array, 'L')
            
            new_width = 256
            new_height = 256
            resized_image = image.resize((new_width, new_height), Image.BILINEAR)
            
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_file_path = os.path.join(image_folder, output_filename)
            resized_image.save(output_file_path)
    
    print('이미지 저장 완료')
    
    return output_file_path
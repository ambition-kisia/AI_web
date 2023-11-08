from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import visualization_docs
import visualization_mal
import resnet_doc
import resnet_mal


app = Flask(__name__)

# delete all files in the directory
def delete_all_files_in_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) and filename != 'test.png':
                    os.remove(file_path)
                    print(f"{file_path} 삭제됨")
            print("디렉토리 내의 모든 파일 삭제 완료")
        else:
            print("디렉토리가 존재하지 않습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
 
file_folder = "uploaded_file/"
image_folder = "static/images/"
#docs = ['.xlsx', '.xls', '.txt', '.pptx', '.ppt', '.pdf', '.hwp', '.docx', '.doc'] 
        
@app.route('/', methods = ['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        # delete file if they exist
        delete_all_files_in_directory(file_folder)
        delete_all_files_in_directory(image_folder)
        
        # save the file
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(file_folder + filename)
        
        # visualization and predict
        if os.path.splitext(filename)[1] == '.byte':
            get_image = visualization_mal.visualization()
            predict = resnet_mal.predict()
            
        else:
            get_image = visualization_docs.visualization()
            predict = resnet_doc.predict()
        
        image_name = os.path.splitext(filename)[0] + '.png'
          
        return render_template('get_result.html', image=get_image, image_name = image_name, result = predict)
    
    else:
        return render_template('file_upload.html')
    
    
if __name__ == '__main__':
    app.run(port=5001, debug=True)
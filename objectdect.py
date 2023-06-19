from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('obj1.html')

@app.route('/process', methods=['POST'])
def process():
    
    image_file = request.files['image']
    txt_file = request.files['txt']

    
    txt_content = txt_file.read().decode('utf-8')

    
    image_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    
    image_height, image_width, _ = image.shape

    
    bounding_boxes = []
    for line in txt_content.splitlines():
        values = line.strip().split(' ')
        obj_id, x_norm, y_norm, width_norm, height_norm = values

        
        obj_id = int(obj_id)
        x_norm, y_norm, width_norm, height_norm = float(x_norm), float(y_norm), float(width_norm), float(height_norm)

        
        x = int((x_norm - width_norm / 2) * image_width)
        y = int((y_norm - height_norm / 2) * image_height)
        width = int(width_norm * image_width)
        height = int(height_norm * image_height)

        
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        
        bbox = {'id': obj_id, 'x': x, 'y': y, 'width': width, 'height': height}
        bounding_boxes.append(bbox)

    
    _, processed_image = cv2.imencode('.jpg', image)
    processed_image_base64 = base64.b64encode(processed_image).decode('utf-8')

    return render_template('obj2.html', image_data=processed_image_base64, bounding_boxes=bounding_boxes)

if __name__ == '__main__':
    app.run(debug=True)

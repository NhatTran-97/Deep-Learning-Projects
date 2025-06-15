import cv2
import numpy as np
import torch
import torch.nn as nn
from  models import SimpleCNN
from scipy.special import softmax
import onnxruntime as ort

class Detect_Model(object):
    #categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    def __init__(self, size,weights=None):
        self.size = size
        if weights is not None:
            self.weights_extension = weights.split(".")[-1]; 
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    def prepare_data(self,frame):
        data_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        data_image = cv2.resize(data_image,(self.size, self.size))
        data_image = np.transpose(data_image, (2,0,1)) / 255.0
        data_image = data_image[None, :, :, :].astype(np.float32)

        if self.weights_extension == 'pt': 
            data_image = torch.from_numpy(data_image).float()
        return data_image
    
class Pytorch_Inference(Detect_Model):
    def __init__(self,model_weights,num_class,size):
        super().__init__(size,model_weights)
        self.num_class = int(num_class)
        self.model_weights = model_weights

        self.softmax = nn.Softmax()
        self.model_inittialize()

    def model_inittialize(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        try:
            self.model = SimpleCNN(num_classes=self.num_class).to(self.device)

            if  self.model_weights.endswith('.pt'):
                model_weights = torch.load(self.model_weights)
                self.model.load_state_dict(model_weights["model"])
                self.model.eval()
            else:
                raise Exception(f"format {self.model_weights} không hợp lệ, cần format .pt")
        except Exception as e:
            print("Lỗi khi tải model_weights:", str(e))

    def prepare_data(self, frame):
        return super().prepare_data(frame)
    
    def run(self, frame):
        try:
            with torch.no_grad():
                output = self.model(self.prepare_data(frame).to(self.device))
                probs = torch.nn.functional.softmax(output, dim=1)
                index_obj = torch.argmax(probs)
                predicted_class = self.categories[index_obj]
            return predicted_class
        
        except Exception as e:
            print(f"Error: {str(e)}")

    
class Onnx_Inference(Detect_Model):
    def __init__(self,model_weights,num_class,size):
        super().__init__(size,model_weights)
        self.num_class = int(num_class)
        self.model_weights = model_weights

        self.model_inittialize()

    def model_inittialize(self):
        try:
            if self.model_weights.endswith('.onnx'):
                self.ort_sess = ort.InferenceSession(self.model_weights, providers=['CUDAExecutionProvider',
                                                                                    'CPUExecutionProvider'])
                self.input_name = self.ort_sess.get_inputs()[0].name
                self.output_name = self.ort_sess.get_outputs()[0].name
            else:
                raise Exception(f"format {self.model_weights} không hợp lệ, cần format .onnx")
        except Exception as e:
            print("Lỗi khi tải model_weights: ", str(e))


    def prepare_data(self, frame):
        return super().prepare_data(frame)
    
    def run(self, frame):
        try:
            if self.ort_sess is None:
                #print("Lỗi: ort_sess chưa được khởi tạo. Hãy chắc chắn rằng model_inittialize đã được gọi trước.")
                #raise  AttributeError("ort_sess chưa được khởi tạo")
                return None
            else:
                outputs = self.ort_sess.run([self.output_name], {self.input_name: self.prepare_data(frame)})
                outputs = np.squeeze(outputs)
                index_obj = np.argmax(softmax(outputs))
                predicted_class = self.categories[index_obj]
                return predicted_class
        except Exception as e:
            print(f"Error: {str(e)}")

class Animal_Detect():
    def __init__(self, model_weights, num_class, size ):
        self.model_weights = model_weights
        self.num_class = num_class
        self.size = size

        self.Model_Initialization()
    
    def Model_Initialization(self):
        try:
            if self.model_weights.endswith('.pt'):
                self.inference_model  = Pytorch_Inference(self.model_weights, self.num_class, self.size)
            elif self.model_weights.endswith('.onnx'):
                self.inference_model  = Onnx_Inference(self.model_weights, self.num_class, self.size)
            else:
                raise ValueError("Định dạng weights không hơp lệ")
            
        except Exception as e:
            print(f"lỗi khi khởi tạo model: ", str(e))
    
    def run(self, frame):
        return self.inference_model.run(frame)

if __name__ == "__main__":

    data = cv2.imread("inferences/5.jpg")

    # prepare = PrepareData(size=224,weights="animal.pt")
    # output = prepare.prepare_data(data)
    # print("test: ", type(output))
    # print(output.shape)



    # IR_torch = Pytorch_Inference("animal.pt",10,224)
    # output = IR_torch.run(data)
    # print("output: ", output)

    IR_ONNX = Onnx_Inference("animal.pt",10,224)
    output = IR_ONNX.run(data)
    print(output)
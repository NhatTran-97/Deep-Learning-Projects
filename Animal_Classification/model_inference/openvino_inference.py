class Openvino_inference(prepareData):
    def __init__(self, weights, num_classes, size):
        super().__init__(size, weights)
        self.size = size
        self.num_classes = int(num_classes)
        self.model_weights = weights
        self.model_initialize()

    def model_initialize(self):
        try:
            if self.model_weights.endswith('.xml'):
                self.core = ov.Core()
                self.openvino_session = self.core.compile_model(self.model_weights, "AUTO")
                self.openvino_session = self.openvino_session.create_infer_request()
            else:
                raise Exception(f"format {self.model_weights} không hợp lệ, cần format .xml")
        except Exception as e:
            print("Lỗi khi tải model_weights: ", str(e))

    def prepareData(self, image):
        return super().prepare_data(image)
    
    def run(self, image):
        try:
            if self.openvino_session is None:
                #print("Lỗi: ort_sess chưa được khởi tạo. Hãy chắc chắn rằng model_inittialize đã được gọi trước.")
                #raise  AttributeError("ort_sess chưa được khởi tạo")
                return None
            else:
                input_tensor = ov.Tensor(array = self.prepareData(image))
                self.openvino_session.set_input_tensor(input_tensor)
                self.openvino_session.start_async()
                self.openvino_session.wait()
                outputs = self.openvino_session.get_output_tensor()
                output_buffer = outputs.data
                index_obj = np.argmax(output_buffer)
                predicted_class = self.categories[index_obj]
                return predicted_class
        except Exception as e:
            print(f"Error: {str(e)}")

def get_args():
    parser = ArgumentParser(description="Pytorch model")
    # self.parser.add_argument("--epochs", type = int, default=10, help="Number of epochs")
    # self.parser.add_argument("--root", type = str, default="data", help="root of dataset")
    parser.add_argument("--weights", type = str, default="animal.xml", help="path of pytorch model")
    parser.add_argument("--source", "-s",type=str, default="8.jpg", help="image or video")
    # parser.add_argument("--imgPath", type = str, default="8.jpg", help="source of img")
    parser.add_argument("--conf", type = float, default= '0.75', help="Confidence score")
    parser.add_argument("--imgSize", type = int, default="416", help="Image size")
    parser.add_argument("--device", type = str, default="cpu", help="Device")
    parser.add_argument("--num_class",type=int, default=10, help="Image Size")
    args = parser.parse_args()
    # print("args: ", self.args)
    return args

if __name__ == "__main__":
#     # path = "animal.onnx
    args = get_args()
    openvino_Inference = Openvino_inference(args.weights, args.num_class, args.imgSize )
    try:
        if args.source.isnumeric():
            print(args.source)
            cap = cv2.VideoCapture(int(args.source),cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise Exception("Không thể mở camera")
            print("Mở camera thành công")

            while True:
                ret, current_frame = cap.read()
                if not ret:
                    print("Không thể đọc được frame từ camera.")
                    break
                #output_classifier = onnx_IR.run(current_frame)
                #output_classifier = torch_IR.run(current_frame)

                output_classifier = openvino_Inference.run(current_frame)
                cv2.putText(current_frame,output_classifier, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                                    
                cv2.imshow("image",current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát.
                    break
        else:
            img_frame = cv2.imread(args.source)
            if img_frame is None:
                raise Exception(f"Không thể đọc file hình ảnh")
            
            print("đọc file hình ảnh thành công")

            #output_classifier = onnx_IR.run(img_frame)
            #output_classifier = torch_IR.run(img_frame)
            output_classifier = openvino_Inference.run(img_frame)
            cv2.putText(img_frame,output_classifier, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            cv2.imshow("image", img_frame)
            cv2.waitKey(0)
     
    except Exception as e:
        print("Error ", e)
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
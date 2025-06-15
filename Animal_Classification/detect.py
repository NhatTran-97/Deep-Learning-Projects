import cv2
from argparse import ArgumentParser
from inference_model import Pytorch_Inference,Onnx_Inference,Animal_Detect

def get_args():
    parser = ArgumentParser(description="CNN Inference")
    parser.add_argument("--source", "-s",type=str, default="inferences/5.jpg", help="image or video") #  # file/folder, 0 for webcam
    parser.add_argument("--image_size", "-i",type=int, default=224, help="Image Size")
    parser.add_argument("--num_class",type=int, default=10, help="Image Size")
    parser.add_argument("--weights","-c",type=str, default="animal.pt")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    #torch_IR = Pytorch_Inference(args.weights, args.num_class, args.image_size)
    #onnx_IR = Onnx_Inference(args.weights,args.num_class,args.image_size)

    animal_IR = Animal_Detect(args.weights, args.num_class, args.image_size)
    

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

                output_classifier = animal_IR.run(current_frame)
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
            output_classifier = animal_IR.run(img_frame)
            cv2.putText(img_frame,output_classifier, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            cv2.imshow("image", img_frame)
            cv2.waitKey(0)
     
    except Exception as e:
        print("Error ", e)
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()



"""
python main_detect.py -s 0
"""
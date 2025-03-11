pip install openvino-dev tensorflow
omz_downloader --name yolo-v4-tiny-tf
omz_converter --name yolo-v4-tiny-tf
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/refs/heads/master/data/dataset_classes/coco_80cl.txt
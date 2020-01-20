# IoT_YOLO_object_detection

Security application using the YOLO object detector. 

Frames are recorded on a RPi and sent to the server that applies YOLO if motion is detected. YOLO is computationally demanding, which is why processing is done server side. 

## Usage

### Client

```
python client.py --server-ip SERVER_IP
```

### Server 

```
python server.py --conf config/config.json
```

## Install Notes

* imagezmq is as of now not available on PyPI - get it by cloning the [repo](https://github.com/jeffbass/imagezmq
)
* [YOLOv3](https://pjreddie.com/darknet/yolo/) weights needs to be downloaded and placed in the yolo-coco folder
 
## Acknowledgements

* [Adrian Rosebrock - Pyimagesearch](https://www.pyimagesearch.com/) for providing a superb toolkit that helps with the implementation of any project
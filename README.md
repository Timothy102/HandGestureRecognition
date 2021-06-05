# Building a HandGestureRecognition Neural Network Inference Model with Raspberry PI. 

Real-time recognition of dynamic hand gestures from video streams is a challenging task since (i) there is no indication when a gesture starts and ends in the video, (ii) performed gestures should only be recognized once, and (iii) the entire architecture should be designed considering the memory and power budget. In this work, we address these challenges by proposing a hierarchical structure enabling offline-working convolutional neural network (CNN) architectures to operate online efficiently by using sliding window approach. The proposed architecture consists of two models: (1) A detector which is a lightweight CNN architecture to detect gestures and (2) a classifier which is a deep CNN to classify the detected gestures. In order to evaluate the single-time activations of the detected gestures, we propose to use the Levenshtein distance as an evaluation metric since it can measure misclassifications, multiple detections, and missing detections at the same time. We evaluate our architecture on two publicly available datasets - EgoGesture and NVIDIA Dynamic Hand Gesture Datasets - which require temporal detection and classification of the performed hand gestures. ResNeXt-101 model, which is used as a classifier, achieves the state-of-the-art offline classification accuracy of 94.04% and 83.82% for depth modality on EgoGesture and NVIDIA benchmarks, respectively. In real-time detection and classification, we obtain considerable early detections while achieving performances close to offline operation. The codes and pretrained models used in this work are publicly available.


## Workflow

### Requirements

* TF Lite Interpreter
* A Virtual-Env or the conda enviroment is recommended. Use 


```bash
conda create -n myenv python
conda install -n myenv scipy 

```


or 

```python3
python3 -m pip install --upgrade pip
pip3 install --upgrade virtualenv
virtualenv -p python3 << name >>
``` 

Download the TensorFlow Lite Interpreter from the website. 

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
``` 

The inference part requires the TensorFlow Lite Interpreter alone. The neural network is already compressed in <tflite> format. Here's the Python code to initialize the interpreter.
  
  
``` python3
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path = model_path)  
``` 
  
  
## Inference
  
Simply run the Makefile in your terminal.
  
```golang
make build  
```
 
It is comprised of taking the picture using raspistill and recieving the same image into a python script. 
Once done, the model will print the desired category. Happy codding! :)
  
 
  

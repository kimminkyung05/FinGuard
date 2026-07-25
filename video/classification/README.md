# Classification

XceptionNet from our paper trained on our FaceForensics++ dataset. Besides the full image models, all models were trained on slightly enlarged face crops with a scale factor of 1.3.
The models were trained using the Face2Face face tracker, though the `detect_from_models.py` file uses the freely available dlib face detector.

Note that we provide the trained models from our paper which have not been fine-tuned for general compressed videos. You can find our used models under [this link](http://kaldir.vc.in.tum.de:/FaceForensics/models/faceforensics++_models.zip).   

## Virtual environment setup (Windows PowerShell)

This project requires Python 3.6. From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.6 -m venv .venv36
.\.venv36\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r video/classification/requirements.txt
```

If PowerShell prevents activation, run the following command once in the current terminal and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

When finished, leave the environment with `deactivate`. Do not commit the `.venv36/` directory.

## Run inference

Run detection on a video file or a directory of `.mp4` or `.avi` files:

```shell
python -m video.classification.detect_from_video \
  -i <path to input video or folder of videos> \
  -m <path to model file> \
  -o <path to output folder>
```  
Enable CUDA with `--cuda`, or see all parameters with `python -m video.classification.detect_from_video -h`.



# Requirements

- python 3.6
- requirements.txt

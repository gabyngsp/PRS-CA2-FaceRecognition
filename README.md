# SECTION 1 : PRS-CA2-FaceRecognition
MTech 2019/2020 - Intelligent System - CA2 Face Recognition

![logo](resources/face.png)

# SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT



# SECTION 3 : CREDITS / PROJECT CONTRIBUTION
| Official Full Name | Student ID (MTech Applicable)| Work Items (Who Did What) | Email (Optional) |
| :---: | :---: | :---: | :---: |
| TEA LEE SENG | A0198538J | Data processing, network design and troubleshoot,project report | e0402079@u.nus.edu / TEALEESENG@gmail.com |
| NG SIEW PHENG | A0198525R  | Data processing, model training and fine-tuning, project report | e0402066@u.nus.edu |
| YANG XIAOYAN| A0056720L | Data processing, model training and fine-tuning, project report | e0401594@u.nus.edu |

# SECTION 4 : 
## Developer Guide

To Train and test CNN network
1. python3 -m venv PRS
2. source PRS/bin/activate
3. git clone https://github.com/gabyngsp/PRS-CA2-FaceRecognition.git
4. cd PRS-CA2-FaceRecognition
5. sudo apt-get install python-opencv build-essential cmake libgtk-3-dev libboost-python-dev pkg-config libx11-dev libatlas-base-dev python3-dev python3-pip
6. pip3 install -r requirements.txt
7. pip3 install tensorflow-gpu # or pip3 install tensorflow #for CPU version!! Really?? 
8. wget http://home.leeseng.tech/training-data-20190925.zip
9. unzip training-data-20190925.zip
10. python3 trainCNN.py
11. python3 testCNN.py

To try video conversion
1. cd PRS-CA2-FaceRecognition
2. wget http://home.leeseng.tech/rawData-20190910.zip
3. unzip rawData-20190910.zip
3. python3 videoToImages.py       # slow due to face clipping.
4. rm data-face data-full         # after review video conversion, unzip training-data-20190925.zip for training.

To Code, 
- runs pycharm community edition. 
- open project on folder, PRS-CA2-FaceRecognition. 
- setup "Project Interpreter with existing VirtualEnv"
- run/debug trainCNN.py
- note: uses pycharm 2019.1.x. pycharm 2019.2.x needs to comments out server.py in pydevd_dont_trace_files.py under pycharm program folder. Refers bug report, https://youtrack.jetbrains.com/issue/PY-37609




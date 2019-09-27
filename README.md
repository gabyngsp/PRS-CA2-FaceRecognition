# SECTION 1 : PRS-CA2-FaceRecognition
MTech 2019/2020 - Intelligent System - CA2 Face Recognition

![logo](resources/face.png)

# SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT



# SECTION 3 : CREDITS / PROJECT CONTRIBUTION
| Official Full Name | Student ID (MTech Applicable)| Work Items (Who Did What) | Email (Optional) |
| :---: | :---: | :---: | :---: |
| TEA LEE SENG | A0198538J | Data processing, network design and troubleshoot,project report | e0402079@u.nus.edu / TEALEESENG@gmail.com |
| NG SIEW PHENG | A0198525R  | TODO | e0402066@u.nus.edu |
| YANG XIAOYAN| A0056720L | TODO | e0401594@u.nus.edu |

# SECTION 4 : 
## Developer Guide

To run Webhook server.
1. python3 -m venv IRS
2. source IRS/bin/activate
3. git clone https://github.com/XiaoyanYang2008/IRS-CGC-SGEventsFinderChatbot
4. cd IRS-CGC-SGEventsFinderChatbot
5. pip3 install -r webapp/requirements.txt
6. cd webapp/
7. python3 server.py
8. In another terminal, ./ngrok http 5001 
   for Dialogflow Fulfillment Webhook url. Take https ngrok URL as Google Assistant demand https channel. 
9. To debug, 
    - kill server.py at step 7, and 
    - runs pycharm community edition. 
    - open project on folder, IRS-CGC-SGEventsFinderChatbot. 
    - Mark webapps folder as Source Root, 
    - setup Project Interpreter with existing VirtualEnv" 
    - debug server.py
    - note: uses pycharm 2019.1.x. pycharm 2019.2.x needs to comments out server.py in pydevd_dont_trace_files.py under pycharm program folder. Refers bug report, https://youtrack.jetbrains.com/issue/PY-37609


To run Dialogflow agent,
1. create a new Dialogflow agent.
2. import ISS-Singapore-Events-Finder-xxxxx.zip into agent.
3. Update Fulfillment Webhook url. Takes https URL.
4. on Google Assistant, says "Talk to my test app"




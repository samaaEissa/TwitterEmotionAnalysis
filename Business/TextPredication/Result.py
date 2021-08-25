class Predication:
     #Constructor
    def __init__(self,txt,status,startTime=None,endTime=None):  
        self.status=status
        self.txt=txt
        self.startTime=startTime
        self.endTime=endTime
        

class Segment:
     #Constructor
    def __init__(self,txt,startTime=None,endTime=None,words=None):  
        self.txt=txt
        self.startTime=startTime
        self.endTime=endTime
        self.words=words
        

class VideoAnalysis:
     #Constructor
    def __init__(self,txt,txt_status,startTime=None,endTime=None,emotion_statistics={}):  
        self.txt=txt
        self.txt_status=txt_status
        self.startTime=startTime
        self.endTime=endTime
        self.emotion_statistics=emotion_statistics
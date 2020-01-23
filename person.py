class person:
    def __init__(self,eye_feat,face_feat,mouth_feat,nose_feat):
        self.face_feat=[face_feat,eye_feat,mouth_feat,nose_feat]
    def attrchk(self):
        for i in self.face_feat:
            if i == None:
                del self
    def __del__(self):
        print("STFU")
    def __str__(self):
        return self.face_feat
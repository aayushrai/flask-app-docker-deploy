    
#import essentials
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime
import os


class ResoluteaiFaceRecog:
    def __init__(self,BASE_DIR,MODEL_DIR,UNKNOWN_DIR,UNKNOWN_MODEL_DIR,USER_PIC_DIR) :
        self.BASE_DIR = BASE_DIR 
        self.MODEL_DIR = MODEL_DIR  
        self.UNKNOWN_DIR = UNKNOWN_DIR  
        self.UNKNOWN_MODEL_DIR = UNKNOWN_MODEL_DIR 
        self.USER_PIC_DIR = USER_PIC_DIR
        #load existing models
        
        try:
            with (open(os.path.join(MODEL_DIR,"encodings.pickle"), "rb")) as openfile:
                while True:
                    try:
                        self.users_data = pickle.load(openfile)
                        self.known_face_encodings = self.users_data["encodings"]
                        self.known_face_names  = self.users_data["names"]
                    except EOFError:
                        break
        except:
            self.known_face_encodings = []
            self.known_face_names  = []

        try:
            with (open(os.path.join(self.UNKNOWN_MODEL_DIR,"unknown_encodings.pickle"), "rb")) as openfile:
                while True:
                    try:
                        unknown_users_data = pickle.load(openfile)
                        self.unknown_face_encodings = unknown_users_data["encodings"]
                        self.unknown_face_names  = unknown_users_data["names"]
                    except EOFError:
                        break
        except:
            self.unknown_face_encodings = []
            self.unknown_face_names = []        
                    
            
    def recognise_face(self,roi, faces,frame,get_unknowns,recog_threshold):
        """this function checks the faces and returns detected users
        face_recognition api is used here"""
        face_locations = []
        face_locations.append(faces)
        #print(face_locations)
        face_encodings = face_recognition.face_encodings(roi, face_locations)
        face_names = []
        det_user=[]
        if self.known_face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0])
            #commpare if all value of match is true
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
                
            if matches[best_match_index] and face_distances[best_match_index]<= recog_threshold:
                name = self.known_face_names[best_match_index]
                #print(name)              #Check: detected users
                return name
            else:
                unknown_user = "Unknown"
                if get_unknowns== True:
                    unknown_user = self.handle_unknowns(frame, faces, face_encodings,recog_threshold)
                #cv2.imwrite(f_name, u_roi)
                return unknown_user
        else:
            unknown_user = "Unknown"
            if get_unknowns== True:
                unknown_user = self.handle_unknowns(frame, faces, face_encodings,recog_threshold)
            #cv2.imwrite(f_name, u_roi)
            return unknown_user
            

    def det_recog_engine(self,frame, recog = True,get_unknowns=True,recog_threshold=0.6):
        """This function will detect faces and returns bounding boxes
        if the boolean of recog is set true then detected faces are returned
        with name"""   
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model = 'hog')
        #print("Faces detected, ", len(faces))   #chcek for number of faces detected
        detected_users_list = []
        area = 0
        for face in face_locations:
                #try:
                #print(face, type(face))
                y,width,height,x= face   
                if recog==True:
                    #roi= rgb[y:height, x:width]
                    detected_user = self.recognise_face(rgb, face, frame,get_unknowns,recog_threshold)
                    detected_users_list.append(detected_user)
                    cv2.putText(frame, detected_user, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)    
                # get coordinates
                frame = cv2.rectangle(frame, (x,y), (width, height), (255,0,0), 1)
                """area = (x-width)*(y-height)
                cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)"""
                    #cv2.imshow("recog", frame)   #check for knowing detected faces
                    
                """except:
                    return frame,detected_users_list,area"""
        return frame,detected_users_list,area

    def handle_unknowns(self,roi, faces, face_encodings,recog_threshold):
        if len(self.unknown_face_encodings)>0:
            matches = face_recognition.compare_faces(self.unknown_face_encodings, face_encodings[0])
            #commpare if all value of match is true
            face_distances = face_recognition.face_distance(self.unknown_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index]<= recog_threshold:
                unknown_name = self.unknown_face_names[best_match_index]
                #print(name)              #Check: detected users
                un_dir = os.path.join(self.UNKNOWN_DIR,unknown_name)
                list_dir_len = len(os.listdir(un_dir))
                if list_dir_len<6:
                    if matches[best_match_index] and face_distances[best_match_index]>= (recog_threshold-0.10):
                        self.unknown_face_encodings.append(face_encodings[0])
                        self.unknown_face_names.append(unknown_name)
                        img_name = (os.path.join(un_dir, "%d.jpg"%list_dir_len))
                        cv2.imwrite(img_name, roi)            
                return unknown_name
            else:
                self.unknown_face_encodings.append(face_encodings[0])
                new_unknown = "unknown_%d"%(len(self.unknown_face_names)+1)
                un_dir = os.path.join(self.UNKNOWN_DIR,new_unknown)
                if not os.path.isdir(un_dir):
                    os.mkdir(un_dir)
                    cv2.imwrite(os.path.join(un_dir, "1.jpg"), roi)
                self.unknown_face_names.append(new_unknown)
                unkEncoding = {"encodings":self.unknown_face_encodings,"names":self.unknown_face_names}
                f_u = open(os.path.join(self.UNKNOWN_MODEL_DIR,"unknown_encodings.pickle"), "wb")
                f_u.write(pickle.dumps(unkEncoding))
                f_u.close()
                return self.unknown_face_names[-1]
            
        else:
            self.unknown_face_encodings.append(face_encodings[0])
            self.unknown_face_names.append("unknown_1")
            un_dir = os.path.join(self.UNKNOWN_DIR,"unknown_1")
            if not os.path.isdir(un_dir):
                os.mkdir(un_dir)
                cv2.imwrite(os.path.join(un_dir, "1.jpg"), roi)
            unkEncoding = {"encodings":self.unknown_face_encodings,"names":self.unknown_face_names}
            f_u = open(os.path.join(self.UNKNOWN_MODEL_DIR,"unknown_encodings.pickle"), "wb")
            f_u.write(pickle.dumps(unkEncoding))
            f_u.close()
            
            return self.unknown_face_names[0]

    def retrain_fn(self,new_users=[]):
        start = datetime.now()
        try:
            prev_users_count = len(set(self.users_data['names']))
        except:
            prev_users_count = 0
            
        if len(new_users):
            for users in os.listdir(self.USER_PIC_DIR):
                #get ENcodings
                try:
                    if users in new_users:
                        user_path = os.path.join(self.USER_PIC_DIR,users)
                        for user_image_path in os.listdir(user_path):
                            if user_image_path.endswith(".jpg") or user_image_path.endswith(".jpeg") :
                                image = cv2.imread(os.path.join(user_path,user_image_path))
                                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                # detect the (x, y)-coordinates of the bounding boxes
                                # corresponding to each face in the input image
                                face_locations = face_recognition.face_locations(rgb, model = 'hog')
                                encodings = face_recognition.face_encodings(rgb, face_locations)
                                for encoding in encodings:
                                    if len(encoding)>0:
                                        #update encodings
                                        self.known_face_encodings.append(encoding)
                                        #update user names
                                        self.known_face_names.append(users)
                except:
                    print("Error while retraining for new users.....")
        else:
            # initialize the list of known encodings and known names
            self.known_face_encodings = []
            self.known_face_names = []
            #get names and update encoding lists
            for users in os.listdir(self.USER_PIC_DIR):
                #get ENcodings
                try:
                    user_path = os.path.join(self.USER_PIC_DIR,users)
                    for user_image_path in os.listdir(user_path):
                        if user_image_path.endswith(".jpg") or user_image_path.endswith(".jpeg") :
                            image = cv2.imread(os.path.join(user_path,user_image_path))
                            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            # detect the (x, y)-coordinates of the bounding boxes
                            # corresponding to each face in the input image
                            face_locations = face_recognition.face_locations(rgb, model = 'hog')
                            encodings = face_recognition.face_encodings(rgb, face_locations)
                            for encoding in encodings:
                                if len(encoding)>0:
                                    #update encodings
                                    self.known_face_encodings.append(encoding)
                                    #update user names
                                    self.known_face_names.append(users)
                except:
                    print("Error while retraining.....")

    

        reg_users_face_encodings = {"encodings": self.known_face_encodings, "names":  self.known_face_names}
        filename_path = os.path.join(self.MODEL_DIR,"encodings.pickle")
        f = open(filename_path, "wb")
        f.write(pickle.dumps(reg_users_face_encodings))
        f.close()


        updated_users_count = len(set(reg_users_face_encodings['names']))
        print("{0} New users added".format(updated_users_count-prev_users_count))
        print("Retrained {0} users and updated {1} encodings in {2}".format(updated_users_count, len(self.known_face_encodings), datetime.now()-start))

    
    def reset_user(self,user_name):
        user_path = os.path.join(self.USER_PIC_DIR,user_name)
        removed_pictures_count = 0
        if len(os.listdir(user_path))>=1:
            for pictures in os.listdir(user_path):
                os.remove(os.path.join(user_path,pictures))
                print(os.path.join(user_path,pictures))
                removed_pictures_count+=1
        print("Removed %d images from user %s"%(removed_pictures_count, user_name))
        
    
    def remove_user(self,user_name):
        user_path = os.path.join(self.USER_PIC_DIR,user_name)
        if os.path.isdir(user_path):
            os.rmdir(user_path)
            pass
        
        print("Successfully removed %s from database"%(user_name))




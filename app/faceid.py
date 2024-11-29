# Import Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import Other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


# Build App and layout
class CamApp(App):
    def build(self):
        # main components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text='Verify', size_hint=(1,.1), on_press=self.verify)
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,.1))

        # Add Items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5',custom_objects={'L1Dist':L1Dist})

        # setup video capture device
        self.capture = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    # Run continously to get webcam feed
    def update(self,*args):
        
        # Read Image from opencv
        ret, frame = self.capture.read()
        frame = frame[320:320+250, 450:450+250, :]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    def preprocess(self, file_path): 
         # Read image from file path 
         byte_img = tf.io.read_file(file_path)
         # Load in the Image
         img = tf.io.decode_jpeg(byte_img)
         # Resizing the Image to be 100*100
         img = tf.image.resize(img, (100,100))
         # Scale Image between 0 and 1 
         img = img/255.0
         return img
    
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture Image from web cam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')

         # Read Image from opencv
        ret, frame = self.capture.read()
        frame = frame[320:320+250, 450:450+250, :]
        cv2.imwrite(SAVE_PATH, frame)
         

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            if image.startswith('.'):
                print(f"Skipping unsupported or hidden file: {image}")
                continue
                
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        print('reached over here 😁', verified)

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))
        
        return results, verified

if __name__ == '__main__':
    CamApp().run()

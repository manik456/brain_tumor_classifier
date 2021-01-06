import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.title('Brain \U0001F9E0 Tumor Detector ')

st.subheader('Find out whether there is a tumor \U0001F534 in the brain (or) \
         not \U0001F7E2 by uploading the MRI \U0001F4C1 of it ')
          

class_labels={0:'No \U0001F7E2',1:'a \U0001F534'}

#st.subheader('Upload Brain MRI'+'\U0001F4C1')

st.write('Find some MRI images here : https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection')

inp_t = st.file_uploader(label='Upload MRI here',accept_multiple_files=True)

#load image
@st.cache(show_spinner=False)
def load_img(path):
        ## reading file object and making it to pil image and to np array
        img_l=[]
        for i in path:
                img_byte=i.read()
                img=Image.open(io.BytesIO(img_byte))
                img=img.resize((256,256),Image.ANTIALIAS)
                if img.mode!='L':
                        img=img.convert('L')
                img_arr=np.array(img,dtype='float32')/255
                img_arr=np.expand_dims(img_arr,axis=-1)
                img_l.append(img_arr)
        img=np.stack(img_l)
        return img

## prediction
@st.cache(show_spinner=False)
def pred(img):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path = r'tumor_lite_model.tflite')

    # setting input size
    interpreter.resize_tensor_input(0, [img.shape[0],256,256,1], strict=True)
    interpreter.allocate_tensors()
    #interpreter = load_model()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]['shape']

    input_data = img
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    # making predictions
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    #tflite_resluts = (tflite_results)

    tflite_results = np.concatenate(tflite_results)  # merging all sub arrays
    
    tf_results=[1 if i>0.5 else 0 for i in tflite_results]  # scaling predictions to 0,1.
 
    return tflite_results,tf_results

vis_img = st.sidebar.checkbox('Show Uploaded Images')

# if file is uploaded
if inp_t:
        img = load_img(inp_t)
        
        st.warning('** Uploaded {} images [View images in side Panel]'.format(img.shape[0]))
         
        res_prob,res = np.array(pred(img))  # convert predictions list to array

        fig,ax=plt.subplots()
          
        for i in range(len(res)):
                
                if res[i] == 0:
                  pred_conf = (0.5 - res_prob[i]) / 0.5
                  pred_conf = pred_conf * 100
                else:
                  pred_conf = res_prob[i] * 100
                  
                st.subheader("*Image "+str(i+1)+" : Model predicts there is {}  tumor with [{} % confidence].*".format(class_labels[res[i]],round(pred_conf,2)))
                
                #if st.checkbox('View Image - ' +str(i+1)):
                           #st.image(img[i],use_column_width=True)
                st.write('\n')
                    
                if vis_img:
                        st.sidebar.write('{} - Image Dimensions: {}'.format(str(i+1),img[i].shape))
                        st.sidebar.image(img[i],use_column_width=True)
        st.markdown('---')
        st.error('Dont conclude by looking at predictions, just take them as a reference!!')

## prints model arch flow chart
#if st.sidebar.checkbox('Model Architecture'):            
#        st.sidebar.write(model)
        



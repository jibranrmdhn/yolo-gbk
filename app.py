
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO

model = YOLO("best.pt")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Prediksi YOLO
    results = model(img)
    res_plotted = results[0].plot()
    
    return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

st.title("Gunting Batu Kertas Real-time")
webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Membuat proses lebih lancar)
)
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
from moviepy.editor import VideoFileClip
from moviepy.callbacks import ProgBarLogger
import time

# Import our custom lane detection functions
from lane_detector import (
    DEFAULT_CONFIG,
    frame_processor_detailed,
    frame_processor
)

# ---------------------------------------------------------
# Streamlit Page Configurations & Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Driving: Real-Time Road Lane Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background: linear-gradient(135deg, #0d0f12 0%, #151922 100%);
        color: #e2e8f0;
    }
    
    /* Headers styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #38bdf8 !important;
        font-weight: 600;
    }
    
    /* Custom container styling */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.3);
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.4) rgba(56, 189, 248, 0.4) transparent rgba(56, 189, 248, 0.4) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0d10 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #38bdf8, #3b82f6);
    }
    
    /* Status indicators */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    .badge-info {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration - Interactive CV Parameters
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Parameters</h2>", unsafe_allow_html=True)
st.sidebar.write("Fine-tune the computer vision pipeline parameters in real-time.")

# 1. Image Preprocessing (Blur & Edges)
with st.sidebar.expander("🔍 Image Preprocessing", expanded=True):
    kernel_size = st.slider("Gaussian Kernel Size", min_value=3, max_value=15, value=DEFAULT_CONFIG["kernel_size"], step=2, 
                            help="Larger values blur the image more to reduce high-frequency noise.")
    low_threshold = st.slider("Canny Low Threshold", min_value=0, max_value=255, value=DEFAULT_CONFIG["low_threshold"],
                              help="Edges with intensity gradients below this are discarded.")
    high_threshold = st.slider("Canny High Threshold", min_value=0, max_value=255, value=DEFAULT_CONFIG["high_threshold"],
                               help="Edges with intensity gradients above this are kept.")

# 2. Region of Interest (ROI) Trapezoid Mask
with st.sidebar.expander("📐 Region of Interest (ROI)", expanded=False):
    st.write("Adjust vertices as a fraction of the image height/width:")
    roi_bottom_left = st.slider("Bottom-Left X", min_value=0.0, max_value=0.5, value=DEFAULT_CONFIG["roi_bottom_left"], step=0.01)
    roi_top_left = st.slider("Top-Left X", min_value=0.2, max_value=0.5, value=DEFAULT_CONFIG["roi_top_left"], step=0.01)
    roi_top_right = st.slider("Top-Right X", min_value=0.5, max_value=0.8, value=DEFAULT_CONFIG["roi_top_right"], step=0.01)
    roi_bottom_right = st.slider("Bottom-Right X", min_value=0.5, max_value=1.0, value=DEFAULT_CONFIG["roi_bottom_right"], step=0.01)
    roi_top_y = st.slider("Top Y boundary", min_value=0.4, max_value=0.8, value=DEFAULT_CONFIG["roi_top_y"], step=0.01,
                          help="Specifies how far up the road we look.")
    roi_bottom_y = st.slider("Bottom Y boundary", min_value=0.8, max_value=1.0, value=DEFAULT_CONFIG["roi_bottom_y"], step=0.01,
                             help="Usually aligned with the bottom of the dashboard.")

# 3. Hough Line Transform Settings
with st.sidebar.expander("📈 Hough Transform", expanded=False):
    rho = st.slider("Rho (Pixels)", min_value=1, max_value=5, value=DEFAULT_CONFIG["rho"],
                    help="Distance resolution of the accumulator in pixels.")
    theta = st.slider("Theta (Degrees)", min_value=1, max_value=5, value=DEFAULT_CONFIG["theta"],
                      help="Angular resolution of the accumulator in degrees.")
    threshold = st.slider("Threshold (Votes)", min_value=5, max_value=100, value=DEFAULT_CONFIG["threshold"],
                          help="Minimum number of intersections in Hough space to detect a line.")
    min_line_len = st.slider("Min Line Length (Pixels)", min_value=5, max_value=200, value=DEFAULT_CONFIG["min_line_len"],
                             help="Minimum length of a line segment to be accepted.")
    max_line_gap = st.slider("Max Line Gap (Pixels)", min_value=5, max_value=600, value=DEFAULT_CONFIG["max_line_gap"],
                             help="Maximum allowed gap between points on the same line to link them.")

# 4. Rendering Options
with st.sidebar.expander("🎨 Rendering", expanded=False):
    lane_color = st.color_picker("Lane Line Color", value="#FF0000", help="Choose overlay color.")
    lane_thickness = st.slider("Line Thickness", min_value=2, max_value=30, value=DEFAULT_CONFIG["lane_thickness"])

# Convert Hex to RGB
lane_color_rgb = list(int(lane_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# Create config dictionary from parameters
config = {
    "kernel_size": kernel_size,
    "low_threshold": low_threshold,
    "high_threshold": high_threshold,
    "rho": rho,
    "theta": theta,
    "threshold": threshold,
    "min_line_len": min_line_len,
    "max_line_gap": max_line_gap,
    "roi_bottom_left": roi_bottom_left,
    "roi_top_left": roi_top_left,
    "roi_top_right": roi_top_right,
    "roi_bottom_right": roi_bottom_right,
    "roi_top_y": roi_top_y,
    "roi_bottom_y": roi_bottom_y,
    "lane_color_r": lane_color_rgb[0],
    "lane_color_g": lane_color_rgb[1],
    "lane_color_b": lane_color_rgb[2],
    "lane_thickness": lane_thickness
}

# ---------------------------------------------------------
# MoviePy custom progress logger for Streamlit
# ---------------------------------------------------------
class StreamlitMoviePyLogger(ProgBarLogger):
    def __init__(self, progress_bar, status_text):
        super().__init__(init_state=None, bars=None, logged_bars=None)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_frames = 0
        
    def callback(self, **changes):
        pass
        
    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == 't':
            total = self.state[bar]['total']
            current = value
            self.total_frames = total
            if total > 0:
                pct = float(current) / float(total)
                self.progress_bar.progress(min(pct, 1.0))
                self.status_text.write(f"⚙️ **Processing frame:** `{current}` / `{total}` ({int(pct*100)}%)")

# ---------------------------------------------------------
# Main Page UI Layout
# ---------------------------------------------------------
st.markdown("<h1>🚗 Real-Time Road Lane Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4>Deep Learning & Computer Vision for Autonomous Driving Systems</h4>", unsafe_allow_html=True)
st.write("---")

# Informative Introduction Card
st.markdown("""
<div class='metric-card'>
    <strong>Lane Detection</strong> is a foundational component of modern ADAS (Advanced Driver Assistance Systems) 
    and Self-Driving Cars. By locating boundary markings on the road, an autonomous system can compute the vehicle's position 
    relative to the lane, maintain center alignment, and execute safe lane changes. 
    This application simulates and visualizes the classical edge-based lane detection pipeline.
</div>
""", unsafe_allow_html=True)

# Select mode: Image or Video
mode = st.radio("Select Input Mode:", ["🖼️ Static Image Tuning", "🎥 Video Lane Detection"], horizontal=True)

# Define sample video path and ensure it exists (for cloud deployments)
SAMPLE_VIDEO_DIR = "data"
SAMPLE_VIDEO_PATH = os.path.join(SAMPLE_VIDEO_DIR, "solidWhiteRight.mp4")

if not os.path.exists(SAMPLE_VIDEO_PATH):
    os.makedirs(SAMPLE_VIDEO_DIR, exist_ok=True)
    with st.spinner("Downloading sample video dataset..."):
        try:
            import requests
            url = "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4"
            r = requests.get(url, stream=True)
            with open(SAMPLE_VIDEO_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download sample video: {e}")

# ---------------------------------------------------------
# Mode 1: Static Image Tuning (Educational Step-by-Step)
# ---------------------------------------------------------
if mode == "🖼️ Static Image Tuning":
    st.subheader("Image Pipeline Inspection")
    st.write("Upload a custom road image or inspect using a sample frame from our default video dataset.")
    
    uploaded_image = st.file_uploader("Upload an Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    # Load image source
    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image)
        raw_img = np.array(pil_img)
        # Handle alpha channel
        if raw_img.shape[2] == 4:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2RGB)
    else:
        # Load sample frame from default video
        if os.path.exists(SAMPLE_VIDEO_PATH):
            cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
            ret, frame = cap.read()
            if ret:
                # opencv reads BGR, convert to RGB
                raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raw_img = np.zeros((540, 960, 3), dtype=np.uint8)
                st.warning("Failed to extract frame from sample video.")
            cap.release()
        else:
            raw_img = np.zeros((540, 960, 3), dtype=np.uint8)
            st.error(f"Sample video not found at {SAMPLE_VIDEO_PATH}. Please upload a custom image.")
            
    # Process the image
    with st.spinner("Processing image steps..."):
        stages = frame_processor_detailed(raw_img, config)
        
    # Visual Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. Original Frame",
        "2. Grayscale & Blur",
        "3. Canny Edges",
        "4. ROI Boundary",
        "5. Masked ROI",
        "6. Detected Hough Lines",
        "7. Final Lane Overlay"
    ])
    
    with tab1:
        st.write("**Original Road Camera Input**")
        st.image(stages["original"], use_column_width=True)
        st.markdown("""
        * **Input:** 3-channel RGB image captured by vehicle-mounted cameras.
        * **Goal:** Extract road features without noise or distortions.
        """)
        
    with tab2:
        col_gray, col_blur = st.columns(2)
        with col_gray:
            st.write("**Grayscale Conversion**")
            st.image(stages["grayscale"], use_column_width=True)
        with col_blur:
            st.write(f"**Gaussian Blur (Kernel: {config['kernel_size']}x{config['kernel_size']})**")
            st.image(stages["blur"], use_column_width=True)
        st.markdown(f"""
        * **Grayscale**: Reduces calculation depth from 3 channels (RGB) to 1, increasing computational speed.
        * **Gaussian Blur**: Applies a convolution filter using Gaussian distribution to smooth out high-frequency noise. 
          *Current kernel size is set to `{config['kernel_size']}x{config['kernel_size']}`.*
        """)
        
    with tab3:
        st.write(f"**Canny Edge Detection (Low: {config['low_threshold']}, High: {config['high_threshold']})**")
        st.image(stages["edges"], use_column_width=True)
        st.markdown(f"""
        * **Canny Edge Detection**: Computes gradient intensities across the image. 
        * It uses double-threshold hysteresis to trace edges:
          * Intensity gradients below `{config['low_threshold']}` are discarded.
          * Intensity gradients above `{config['high_threshold']}` are marked as strong edges.
          * Intermediate values are connected only if they connect to a strong edge.
        """)
        
    with tab4:
        st.write("**Region of Interest boundary (Yellow Trapezoid)**")
        st.image(stages["roi_visualizer"], use_column_width=True)
        st.markdown("""
        * **ROI Polygon**: Restricts line search to the road section immediately ahead of the vehicle.
        * Eliminates background clutter (sky, trees, side barriers) which might confuse the line detector.
        """)
        
    with tab5:
        st.write("**Masked Region of Interest (Logical Bitwise AND)**")
        st.image(stages["roi"], use_column_width=True)
        st.markdown("""
        * **Masking**: Applies a bitwise `AND` between the Canny output and the ROI polygon.
        * Only active edges lying within our road lane region are retained.
        """)
        
    with tab6:
        st.write("**Detected Hough Lines (Green Lines)**")
        st.image(stages["hough"], use_column_width=True)
        st.markdown(f"""
        * **Probabilistic Hough Line Transform**: Detects straight lines by voting on intersections in parametric space.
        * Configured parameters:
          * **Rho**: `{config['rho']}` pixel distance resolution.
          * **Theta**: `{config['theta']}` degree angle resolution.
          * **Threshold**: `{config['threshold']}` votes required.
          * **Min Line Length**: `{config['min_line_len']}` pixels.
          * **Max Line Gap**: `{config['max_line_gap']}` pixels.
        """)
        
    with tab7:
        st.write("**Final Output Overlay**")
        st.image(stages["result"], use_column_width=True)
        
        # Display slopes
        left_lane, right_lane = stages["detected_lanes"]
        
        col_l, col_r = st.columns(2)
        with col_l:
            if left_lane:
                st.info(f"↖️ **Left Lane Slope:** `{left_lane[0]:.4f}` | **Intercept:** `{left_lane[1]:.1f}`")
            else:
                st.warning("↖️ **Left Lane:** Not detected (adjust parameters)")
        with col_r:
            if right_lane:
                st.info(f"↗️ **Right Lane Slope:** `{right_lane[0]:.4f}` | **Intercept:** `{right_lane[1]:.1f}`")
            else:
                st.warning("↗️ **Right Lane:** Not detected (adjust parameters)")
                
        st.markdown("""
        * **Averaging & Extrapolation**: Slopes and intercepts from all Hough line segments are categorized.
          * Negative slope indicates the left lane markings.
          * Positive slope indicates the right lane markings.
        * We compute a length-weighted average of slopes/intercepts, then draw complete continuous lanes from the bottom of the frame up to the ROI top horizon boundary.
        """)

# ---------------------------------------------------------
# Mode 2: Video Lane Detection Pipeline
# ---------------------------------------------------------
else:
    st.subheader("Video Processing Pipeline")
    st.write("Process entire driving video clips with the active parameter settings.")
    
    video_source = st.radio("Choose Video Source:", ["Use Default Sample Video", "Upload Custom Video"], horizontal=True)
    
    in_video_path = None
    
    if video_source == "Use Default Sample Video":
        if os.path.exists(SAMPLE_VIDEO_PATH):
            in_video_path = SAMPLE_VIDEO_PATH
            st.video(SAMPLE_VIDEO_PATH)
        else:
            st.error(f"Sample video not found at {SAMPLE_VIDEO_PATH}. Please upload a custom video file.")
    else:
        uploaded_video = st.file_uploader("Upload an MP4/MOV Video File", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            t_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_file.write(uploaded_video.read())
            in_video_path = t_file.name
            st.video(in_video_path)
            
    if in_video_path:
        st.write("---")
        process_btn = st.button("🚀 Start Lane Detection Processing", type="primary")
        
        if process_btn:
            # Outputs
            out_video_dir = tempfile.gettempdir()
            out_video_path = os.path.join(out_video_dir, "processed_output.mp4")
            
            # Setup Progress Indicators
            st.write("### Processing Status")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                # Load video using moviepy
                clip = VideoFileClip(in_video_path)
                
                # Setup logger for moviepy progress updates
                my_logger = StreamlitMoviePyLogger(progress_bar, status_text)
                
                # Apply pipeline to every frame
                processed_clip = clip.fl_image(lambda frame: frame_processor(frame, config))
                
                # Write file
                with st.spinner("Writing video file... This may take a few moments."):
                    processed_clip.write_videofile(
                        out_video_path,
                        audio=False,
                        codec='libx264',
                        logger=my_logger
                    )
                
                # Finalize
                elapsed_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.success(f"🎉 **Success!** Video processed in {elapsed_time:.2f} seconds.")
                
                # Close clips to free memory
                clip.close()
                processed_clip.close()
                
                # Display output video
                st.write("### 🎬 Processed Video Output")
                st.video(out_video_path)
                
                # Download button
                with open(out_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="detected_road_lanes.mp4",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                # Print traceback to stdout for debugging
                import traceback
                traceback.print_exc()

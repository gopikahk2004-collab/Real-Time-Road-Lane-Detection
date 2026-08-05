import numpy as np
import cv2

DEFAULT_CONFIG = {
    "kernel_size": 5,
    "low_threshold": 50,
    "high_threshold": 150,
    "rho": 1,
    "theta": 1,  # in degrees (will be converted to radians np.pi/180)
    "threshold": 20,
    "min_line_len": 20,
    "max_line_gap": 300,
    "roi_bottom_left": 0.10,
    "roi_top_left": 0.40,
    "roi_top_right": 0.60,
    "roi_bottom_right": 0.90,
    "roi_top_y": 0.60,
    "roi_bottom_y": 0.95,
    "lane_color_r": 255,
    "lane_color_g": 0,
    "lane_color_b": 0,
    "lane_thickness": 12
}

def region_selection(image, config=None):
    """
    Determine and mask the region of interest in the input image.
    Parameters:
        image: grayscale or color image.
        config: dictionary of configuration parameters.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    mask = np.zeros_like(image)   
    
    # Check channel count
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    rows, cols = image.shape[:2]
    
    # Calculate vertices using percentage parameters
    bottom_left  = [cols * config["roi_bottom_left"], rows * config["roi_bottom_y"]]
    top_left     = [cols * config["roi_top_left"], rows * config["roi_top_y"]]
    bottom_right = [cols * config["roi_bottom_right"], rows * config["roi_bottom_y"]]
    top_right    = [cols * config["roi_top_right"], rows * config["roi_top_y"]]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, vertices

def hough_transform(image, config=None):
    """
    Apply Probabilistic Hough Transform to find lines.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    rho = config["rho"]
    theta = config["theta"] * np.pi / 180
    threshold = config["threshold"]
    minLineLength = config["min_line_len"]
    maxLineGap = config["max_line_gap"]
    
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines

def average_slope_intercept(lines, slope_min=0.3, slope_max=2.0):
    """
    Find the slope and intercept of the left and right lanes.
    Filters out lines that do not have slopes within [slope_min, slope_max].
    """
    if lines is None:
        return None, None
        
    left_lines    = []  # (slope, intercept)
    left_weights  = []  # (length,)
    right_lines   = []  # (slope, intercept)
    right_weights = []  # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            
            # Check if slope falls within realistic bounds
            if slope_min <= abs(slope) <= slope_max:
                if slope < 0:  # Left lane (slopes up to the right, negative in image coordinates)
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:          # Right lane (slopes down to the right, positive in image coordinates)
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
                    
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts slope and intercept of a line into integer pixel endpoints.
    """
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-5:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines, config=None):
    """
    Find endpoints for full-length left and right lane lines.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    left_lane, right_lane = average_slope_intercept(lines)
    
    rows = image.shape[0]
    y1 = rows * config["roi_bottom_y"]
    y2 = rows * config["roi_top_y"]
    
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=None, thickness=12):
    """
    Draw lane lines onto a blank image overlay and merge with the input image.
    """
    if color is None:
        color = [255, 0, 0]
        
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image, config=None):
    """
    Fast processing function designed to be used with MoviePy's fl_image.
    Processes frame-by-frame using config.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    # convert RGB to grayscale (MoviePy passes RGB frames)
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    k = config["kernel_size"]
    # Ensure kernel size is odd
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(grayscale, (k, k), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur, config["low_threshold"], config["high_threshold"])
    
    # Region Selection
    region, _ = region_selection(edges, config)
    
    # Hough Transform
    hough = hough_transform(region, config)
    
    # Average and draw lane lines
    left_line, right_line = lane_lines(image, hough, config)
    
    lane_color = [config["lane_color_r"], config["lane_color_g"], config["lane_color_b"]]
    result = draw_lane_lines(image, [left_line, right_line], color=lane_color, thickness=config["lane_thickness"])
    return result

def frame_processor_detailed(image, config=None):
    """
    Processes frame and returns a dictionary with all intermediate steps.
    Used for visualization in the Streamlit app.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    # 1. Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Gaussian Blur
    k = config["kernel_size"]
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(grayscale, (k, k), 0)
    
    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, config["low_threshold"], config["high_threshold"])
    
    # 4. Region Selection
    region, vertices = region_selection(edges, config)
    
    # 5. Hough Transform
    hough = hough_transform(region, config)
    
    # Create an image showing all raw Hough lines detected
    hough_lines_img = np.zeros_like(image)
    if hough is not None:
        for line in hough:
            for x1, y1, x2, y2 in line:
                cv2.line(hough_lines_img, (x1, y1), (x2, y2), [0, 255, 0], 2) # Green for individual hough lines
    hough_overlay = cv2.addWeighted(image, 0.8, hough_lines_img, 1.0, 0.0)
    
    # 6. Average and draw lane lines
    left_line, right_line = lane_lines(image, hough, config)
    lane_color = [config["lane_color_r"], config["lane_color_g"], config["lane_color_b"]]
    result = draw_lane_lines(image, [left_line, right_line], color=lane_color, thickness=config["lane_thickness"])
    
    # Convert grayscale/single channel maps to RGB so Streamlit displays them nicely
    grayscale_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    blur_rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
    
    # Draw ROI polygon on a copy of the original image for visual feedback
    roi_visualizer = image.copy()
    cv2.polylines(roi_visualizer, vertices, isClosed=True, color=[0, 255, 255], thickness=3) # Cyan ROI boundary
    
    return {
        "original": image,
        "grayscale": grayscale_rgb,
        "blur": blur_rgb,
        "edges": edges_rgb,
        "roi_visualizer": roi_visualizer,
        "roi": region_rgb,
        "hough": hough_overlay,
        "result": result,
        "detected_lanes": (left_line, right_line)
    }

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="PSR Matrix Organization Quantifier", layout="wide")
st.title("📊 PSR Matrix Organization & Maturation Quantifier")

st.markdown("""
**Important note (methodological scope):**  
This tool quantifies *PSR-based birefringence patterns* as a **proxy for collagen fiber organization and matrix maturation**.  
It does **not** quantify collagen subtypes, molecular abundance, or signal intensity.
""")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Segmentation")
mode = st.sidebar.radio("Threshold Mode", ["Auto+Offset", "Manual"])
offset = st.sidebar.slider("Otsu Offset", -40, 40, -10)
manual_thresh = st.sidebar.slider("Manual Threshold (V)", 0, 255, 110)
sat_min = st.sidebar.slider("Min Saturation (S)", 0, 30, 5)

st.sidebar.header("Hue Ranges (OpenCV HSV)")
red_max = st.sidebar.slider("Red max", 5, 15, 10)
orange_low = st.sidebar.slider("Orange low", 8, 20, 12)
orange_high = st.sidebar.slider("Orange high", 20, 40, 30)
green_low = st.sidebar.slider("Green low", 10, 60, 40)
green_high = st.sidebar.slider("Green high", 60, 120, 90)

st.sidebar.header("Object Filters")
min_length = st.sidebar.slider("Min fiber length (px)", 3, 100, 10)
min_area = st.sidebar.slider("Min area (px²)", 3, 50, 8)

uploaded = st.sidebar.file_uploader(
    "Upload PSR images",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# -------------------------
# ROI selection (Streamlit canvas)
# -------------------------
from PIL import Image
import numpy as np

def draw_polygon_roi(img_rgb):

    # Ensure uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    # Ensure RGB
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=-1)

    if img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    # Convert to PIL
    img_pil = Image.fromarray(img_rgb)

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=img_pil,
        update_streamlit=True,
        height=img_rgb.shape[0],
        width=img_rgb.shape[1],
        drawing_mode="polygon",
        key="roi_canvas",
    )

    return canvas

    if canvas.json_data is None or len(canvas.json_data["objects"]) == 0:
        return None

    try:
        obj = canvas.json_data["objects"][0]
        points = [(p["x"], p["y"]) for p in obj["path"][:-1]]
    except:
        return None

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    return mask


# -------------------------
# Analysis
# -------------------------
def analyze_image(img_rgb, roi_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Threshold
    if mode == "Manual":
        base_mask = v > manual_thresh
    else:
        otsu,_ = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        base_mask = v > np.clip(otsu + offset,0,255)

    sat_mask = s > sat_min
    collagen_mask = base_mask & sat_mask

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    collagen_mask = cv2.morphologyEx(collagen_mask.astype(np.uint8)*255,
                                     cv2.MORPH_OPEN,kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask,cv2.MORPH_CLOSE,kernel)

    # Object filtering
    labels = label(collagen_mask)
    final_mask = np.zeros_like(collagen_mask)
    for r in regionprops(labels):
        if r.major_axis_length >= min_length and r.area >= min_area:
            final_mask[labels == r.label] = 255

    cm = final_mask.astype(bool)

    # Apply ROI
    cm = cm & (roi_mask > 0)

    # Hue classification
    red = (((h <= red_max) | (h >= 170)) & cm)
    orange = ((h >= orange_low) & (h <= orange_high) & cm)
    green = ((h >= green_low) & (h <= green_high) & cm)

    red_px = np.sum(red)
    orange_px = np.sum(orange)
    green_px = np.sum(green)
    total_px = np.sum(cm)

    # Sensitivity analysis
    weights = [0.3, 0.5, 0.7]
    ratios = {}
    for w in weights:
        eff_red = red_px + w * orange_px
        eff_green = green_px + w * orange_px
        ratios[f"RG_ratio_w{w}"] = eff_red / (eff_green + 1e-6)

    maturation_index = ratios["RG_ratio_w0.5"]

    overlay = img_rgb.copy()
    overlay[red] = [255,0,0]
    overlay[orange] = [255,165,0]
    overlay[green] = [0,255,0]

    return {
        "Matrix maturation index (RG 0.5)": maturation_index,
        "RG ratio (w=0.3)": ratios["RG_ratio_w0.3"],
        "RG ratio (w=0.7)": ratios["RG_ratio_w0.7"],
        "Total collagen area (px)": total_px,
        "Red px": red_px,
        "Orange px": orange_px,
        "Green px": green_px,
        "overlay": overlay
    }


# -------------------------
# Main
# -------------------------
if uploaded:
    results = []

    for f in uploaded:
        f.seek(0)
        data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.markdown(f"## ROI for: **{f.name}**")
        roi_mask = draw_polygon_roi(img_rgb)

        if roi_mask is None:
            st.warning("Draw an ROI to continue.")
            st.stop()

        result = analyze_image(img_rgb, roi_mask)
        result["Image"] = f.name
        results.append(result)

    df = pd.DataFrame(results).drop(columns=["overlay"])
    st.subheader("📄 Quantitative results (reviewer-safe)")
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "psr_matrix_maturation_results.csv"
    )

    st.subheader("🔍 Visual QC (structural proxy)")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["overlay"], use_column_width=True)

else:
    st.info("Upload PSR images to begin.")

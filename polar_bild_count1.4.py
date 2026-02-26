import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ---------------------------------------------------------
# Hilfsfunktion: Bild sicher in RGB + uint8 + PIL konvertieren
# ---------------------------------------------------------
def prepare_image(img):
    # Falls float → uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Falls Graustufen → RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Falls RGBA → RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


# ---------------------------------------------------------
# ROI-Zeichnen mit Streamlit Canvas
# ---------------------------------------------------------
def draw_polygon_roi(img_rgb):

    img_rgb = prepare_image(img_rgb)
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


# ---------------------------------------------------------
# Maske aus Canvas extrahieren
# ---------------------------------------------------------
def extract_mask_from_canvas(canvas, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    if canvas.json_data is None:
        return mask

    for obj in canvas.json_data["objects"]:
        if obj["type"] == "polygon":
            points = obj["path"]
            pts = []

            for p in points:
                if p[0] == "L" or p[0] == "M":
                    pts.append([int(p[1]), int(p[2])])

            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.title("Polygon ROI Tool – stabil & fehlerfrei")

uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Bild anzeigen")
    st.image(img_rgb)

    st.subheader("ROI zeichnen")
    canvas = draw_polygon_roi(img_rgb)

    if st.button("Maske erzeugen"):
        mask = extract_mask_from_canvas(canvas, img_rgb.shape)
        st.subheader("ROI Maske")
        st.image(mask, clamp=True)

        st.success("Maske erfolgreich erzeugt!")

import pandas as pd
import streamlit as st
import time
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from bbox_detection.demo_streamlit import YOLOv10Inference
from src.config import PROJECT_ROOT
from src.web_app.bbox_postprocess import merge_vertical_columns
from src.web_app.cloud_modules import get_ocr
from src.web_app.spell_correction import correct_spelling_batch
from src.web_app.merge_articles import main_merge


def yolo_box_detection(image) -> pd.DataFrame:
    """Box detection with YOLO"""
    box_model = YOLOv10Inference(str(PROJECT_ROOT / "bbox_detection" / "doclayout_yolo/models/yolov10/yolov10b_best.pt"))
    annotated_img, bbox_df = box_model.process_image(image)
    st.subheader("Extracted Boxes")
    st.image(annotated_img)
    return bbox_df


# OCR processing functions
def perform_ocr(image, bbox_df) -> pd.DataFrame:
    """Crop image into boxes and perform OCR"""
    merged_boxes = merge_vertical_columns(bbox_df)

    cropped_images = []
    for _, row in merged_boxes.iterrows():
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_image)

    ocr_texts = get_ocr(cropped_images)
    merged_boxes["ocr_text"] = ocr_texts
    return merged_boxes


def correct_spelling(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic spelling correction"""
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
    model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
    model.to("cuda")


    ocr_df["corrected_text"] = correct_spelling_batch(model, tokenizer, ocr_df["ocr_text"].tolist())
    return ocr_df


def merge_into_articles(corrected_df: pd.DataFrame, image: Image) -> str:
    """Organize text into articles with titles"""
    final_text, fig = main_merge(corrected_df, image)
    st.subheader("Organized Articles")
    st.pyplot(fig)
    return final_text


def process_image(uploaded_file):
    """Process the uploaded image through all stages"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Stage 1: Box detection with YOLO
    status_text.text("Stage 1/4: Detecting text boxes with YOLO...")
    image = Image.open(uploaded_file)
    bbox_df = yolo_box_detection(image)
    progress_bar.progress(25)
    time.sleep(0.5)  # For smooth progress bar update

    # Stage 2: OCR
    status_text.text("Stage 2/4: Performing OCR...")
    ocr_df = perform_ocr(image, bbox_df)
    progress_bar.progress(50)
    time.sleep(0.5)

    # Stage 3: Spelling correction
    status_text.text("Stage 3/4: Correcting spelling...")
    corrected_df = correct_spelling(ocr_df)
    progress_bar.progress(75)
    time.sleep(0.5)

    # Stage 4: Merging into articles
    status_text.text("Stage 4/4: Organizing into articles...")
    final_text = merge_into_articles(corrected_df, image)
    progress_bar.progress(100)
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    return final_text


def main():
    st.title("Document OCR Processor")
    st.markdown("""
    Upload a scanned document in JPG format to extract text with OCR.
    The processing includes:
    1. Text box detection (YOLO)
    2. OCR with Yandex Cloud
    3. Spelling correction with Sage T5
    4. Clustering into articles and matching with titles
    """)

    uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Process Document"):
            with st.spinner("Processing your document..."):
                result = process_image(uploaded_file)

            st.success("OCR Processing Complete!")
            st.subheader("Extracted Text (Markdown Format)")
            st.markdown(result)

            # Add download button for the result
            st.download_button(
                label="Download as Markdown",
                data=result,
                file_name="extracted_text.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()
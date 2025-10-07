#!/usr/bin/env python3
from paddleocr import PaddleOCR
import fitz
import cv2
from PIL import Image
import numpy as np
import tqdm
import sys
import logging
import argparse
import pathlib
import io

logging.disable(logging.DEBUG)


def im2stream(im: np.ndarray):
    _, buffer = cv2.imencode(".bmp", im)
    bio = io.BytesIO(buffer)
    return bio


def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    img = fitz.open()
    if args.pure:
        pure = fitz.open()
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
        # use_gpu=True,
    )
    for page_number in tqdm.tqdm(
        range(
            pdf_doc.page_count
            # 1
            # 10
        )
    ):
        page = pdf_doc.load_page(page_number)
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        image_list = page.get_images()
        if not image_list:
            continue
        xref_0 = image_list[0][0]
        bin_img = pdf_doc.extract_image(xref_0)
        img_page = img.new_page(width=page.rect.width, height=page.rect.height)
        xref = img_page.insert_image(
            img_page.rect, stream=bin_img["image"], rotate=page.rotation
        )
        cim = cv2.imdecode(np.frombuffer(bin_img["image"], np.uint8), cv2.IMREAD_COLOR)
        if "Decode" not in pdf_doc.xref_get_keys(xref_0):
            img.xref_set_key(xref, "Decode", "null")
            if pdf_doc.xref_get_key(xref_0, "ColorSpace") == ("name", "/DeviceCMYK"):
                cim = cv2.cvtColor(
                    np.array(
                        Image.fromarray(
                            ~np.array(Image.open(io.BytesIO(bin_img["image"]))),
                            mode="CMYK",
                        ).convert("RGB")
                    ),
                    cv2.COLOR_RGB2BGR,
                )
        else:
            under_control = False
            if pdf_doc.xref_get_key(xref_0, "ColorSpace") == ("name", "/DeviceGray"):
                if pdf_doc.xref_get_key(xref_0, "Decode") == ("array", "[0 1]"):
                    under_control = True
            if not under_control:
                print(f"Warning! Decode presents on page {page_number}! xref:{xref_0}")
                print(pdf_doc.xref_get_key(xref_0, "Decode"))
                print(pdf_doc.xref_get_key(xref_0, "ColorSpace"))
                print(pdf_doc.xref_get_keys(xref_0))
        # I don't know what to do if "Decode" presents
        if page.rotation:
            cim = cv2.rotate(cim, (page.rotation // 90 - 1) % 3)
        if args.cv:
            cv2.imshow(sys.argv[0], cim)
            cv2.waitKey(1)
        if args.no_ocr:
            continue
        if args.pure:
            new_page = pure.new_page(width=page.rect.width, height=page.rect.height)
        text = ocr.predict(cim)
        if not text[0]:
            continue
        t0 = text[0]
        for rec_box, rec_text, rec_score in zip(
            t0["rec_boxes"], t0["rec_texts"], t0["rec_scores"]
        ):

            def position_convert(x):
                return (
                    x[0] / cim.shape[1] * page.rect.width,
                    x[1] / cim.shape[0] * page.rect.height,
                )

            R = fitz.Rect(
                position_convert((rec_box[0], rec_box[1])),
                position_convert((rec_box[2], rec_box[3])),
            )
            if rec_score < 0.9:
                continue
            fn = "helv" if rec_text.isascii() else "china-s"
            fs = R.width / fitz.get_text_length(rec_text, fn, 1)
            img_page.insert_text(
                (R.x0, R.y1),
                rec_text,
                fontname=fn,
                fontsize=fs,
                # render_mode=3,
            )
            if args.pure:
                new_page.insert_text(
                    (R.x0, R.y1),
                    rec_text,
                    fontname=fn,
                    fontsize=fs,
                    render_mode=0,
                )
    if args.cv:
        cv2.destroyAllWindows()
    if args.pure:
        try:
            if pdf_doc.get_page_labels():
                pure.set_page_labels(pdf_doc.get_page_labels())
        except TypeError:
            pass
        try:
            if pdf_doc.get_toc():
                pure.set_toc(pdf_doc.get_toc())
        except TypeError:
            pass
        pure.save(
            pathlib.Path(output_pdf_path).stem + "-pure.pdf", garbage=4, deflate=True
        )
        pure.close()
    try:
        if pdf_doc.get_page_labels():
            img.set_page_labels(pdf_doc.get_page_labels())
    except TypeError:
        pass
    try:
        if pdf_doc.get_toc():
            img.set_toc(pdf_doc.get_toc())
    except TypeError:
        pass
    img.save(output_pdf_path, garbage=4, deflate=True)
    img.close()
    pdf_doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program than adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2025 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("output_file", help="Output PDF file")
    parser.add_argument("-p", "--pure", action="store_true")
    parser.add_argument("-c", "--cv", action="store_true")
    parser.add_argument("-n", "--no-ocr", action="store_true")
    parser.add_argument("-l", "--lang", default="ch")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)

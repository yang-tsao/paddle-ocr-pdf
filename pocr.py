#!/usr/bin/env python3
from paddleocr import PaddleOCR
import fitz
import cv2
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
    ocr = PaddleOCR(use_angle_cls=True, lang=args.lang, use_gpu=True)
    for page_number in tqdm.tqdm(
        range(
            pdf_doc.page_count
            # 16
        )
    ):
        page = pdf_doc.load_page(page_number)
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        image_list = page.get_images()
        if not image_list:
            continue
        pix = fitz.Pixmap(pdf_doc, image_list[0][0])
        cim = cv2.cvtColor(
            np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, pix.n)
            ),
            cv2.COLOR_RGB2BGR,
        )
        cim = cv2.rotate(cim, (page.rotation // 90 - 1) % 3)
        if args.cv:
            cv2.imshow(sys.argv[0], cim)
            cv2.waitKey(1)
        text = ocr.ocr(cim)
        if not text[0]:
            continue
        img_page = img.new_page(width=cim.shape[1], height=cim.shape[0])
        img_page.insert_image(img_page.rect, stream=im2stream(cim))
        if args.pure:
            new_page = pure.new_page(width=cim.shape[1], height=cim.shape[0])
        for o in text[0]:
            R = fitz.Rect(o[0][0], o[0][2])
            word = o[1][0]
            if o[1][1] < 0.9:
                continue
            fn = "helv" if word.isascii() else "china-s"
            fs = R.width / fitz.get_text_length(word, fn, 1)
            img_page.insert_text(
                (R.x0, R.y1),
                word,
                fontname=fn,
                fontsize=fs,
                render_mode=3,
            )
            if args.pure:
                new_page.insert_text(
                    (R.x0, R.y1),
                    word,
                    fontname=fn,
                    fontsize=fs,
                    render_mode=0,
                )
    if args.cv:
        cv2.destroyAllWindows()
    if args.pure:
        pure.set_page_labels(pdf_doc.get_page_labels())
        pure.save(
            pathlib.Path(output_pdf_path).stem + "-pure.pdf", garbage=4, deflate=True
        )
        pure.close()
    img.save(output_pdf_path, garbage=4, deflate=True)
    img.close()
    pdf_doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program than adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2024 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("output_file", help="Output PDF file")
    parser.add_argument("-p", "--pure", action="store_true")
    parser.add_argument("-c", "--cv", action="store_true")
    parser.add_argument("-l", "--lang", default="ch")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)

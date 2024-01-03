#!/usr/bin/python
import fitz  # PyMuPDF
import cv2
import numpy as np
import tqdm
from paddleocr import PaddleOCR
import logging
import argparse

logging.disable(logging.DEBUG)


def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    for page_number in tqdm.tqdm(range(pdf_doc.page_count)):
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
        text = ocr.ocr(cim)

        def conv(C):
            C[0] /= cim.shape[1]
            C[1] /= cim.shape[0]
            C[0] *= page.rect.x1
            C[1] *= page.rect.y1
            return C

        fn = "china-s"
        if not text[0]:
            continue
        for o in text[0]:
            R = fitz.Rect(conv(o[0][0]), conv(o[0][2]))
            word = o[1][0]
            if o[1][1] < 0.9:
                continue
            fs = R.width / fitz.get_text_length(word, fn, 1)
            page.insert_text(
                (R.x0, R.y1),
                word,
                fontname=fn,
                fontsize=fs,
                render_mode=3,
            )
    pdf_doc.save(output_pdf_path, garbage=4, deflate=True)
    pdf_doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program than adds hidden(but copiable) text layer to image pdf.",
        epilog="Copyright (C) 2024 Cao Yang. This is free software; distributed under GPLv3. There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    )
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("output_file", help="Output PDF file")
    args = parser.parse_args()
    process_pdf(args.input_file, args.output_file)

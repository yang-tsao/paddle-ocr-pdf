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


def process_pdf(input_pdf_path, output_pdf_path):
    pdf_doc = fitz.open(input_pdf_path)
    if args.pure:
        pure = fitz.open()
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=args.lang,
        use_gpu=True,
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
        cim = cv2.imdecode(np.frombuffer(bin_img["image"], np.uint8), cv2.IMREAD_COLOR)
        if page.rotation:
            cim = cv2.rotate(cim, (page.rotation // 90 - 1) % 3)
        if args.cv:
            cv2.imshow(sys.argv[0], cim)
            cv2.waitKey(1)
        if args.no_ocr:
            continue
        if args.pure:
            new_page = pure.new_page(width=cim.shape[1], height=cim.shape[0])
        text = ocr.ocr(cim)
        if not text[0]:
            continue
        for o in text[0]:

            def position_convert(x):
                return (
                    x[0] / cim.shape[1] * page.rect.width,
                    x[1] / cim.shape[0] * page.rect.height,
                )

            R = fitz.Rect(
                position_convert(o[0][0]),
                position_convert(o[0][2]),
            )
            word = o[1][0]
            if o[1][1] < 0.9:
                continue
            fn = "helv" if word.isascii() else "china-s"
            fs = R.width / fitz.get_text_length(word, fn, 1)
            page.insert_text(
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
        if pdf_doc.get_page_labels():
            pure.set_page_labels(pdf_doc.get_page_labels())
        if pdf_doc.get_toc():
            pure.set_toc(pdf_doc.get_toc())
        pure.save(
            pathlib.Path(output_pdf_path).stem + "-pure.pdf", garbage=4, deflate=True
        )
        pure.close()

    pdf_doc.save(output_pdf_path, garbage=4, deflate=True)
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
    parser.add_argument("-n", "--no-ocr", action="store_true")
    parser.add_argument("-l", "--lang", default="ch")

    args = parser.parse_args()
    if args.cv:
        cv2.namedWindow(sys.argv[0], cv2.WINDOW_NORMAL)
    process_pdf(args.input_file, args.output_file)

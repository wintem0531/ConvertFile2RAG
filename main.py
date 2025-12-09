import modules.pdfConverter.pdf_service as pdf_service


def main():
    pdf_service.pdf_to_images(
        "test.pdf",
        "test_images",
    )


if __name__ == "__main__":
    main()

import chesspdftofen

def main():
    input_pdf_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.pdf'  # Replace with the path to your input PDF
    output_pdf_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdftofen-master.pdf'  # Replace with the path to your desired output PDF

    # Run the chess board detection and FEN annotation
    for status in chesspdftofen.run(input_pdf_path, output_pdf_path, num_threads=4, num_pages_to_print=10):
        print(status)

if __name__ == "__main__":
    main()

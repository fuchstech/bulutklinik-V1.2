import ftplib
import PyPDF2

from PIL import Image
import pytesseract

# Open the image using PIL
oxy = "Sp: %93"
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class Hastane:
    def __init__(self) -> None:
        self.session = ftplib.FTP('mt-ares.guzelhosting.com','morvecom','226Iry4Tkz')
        self.tahlil = open(r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\Enabiz-Tahlilleri.pdf",'rb')  
        
    def send_tahlil(self):
        self.send_data(self.tahlil,"Kan Tahlili.pdf")
        
    def send_data(self,file,file_name):
        self.session.storbinary(f'STOR /matiricie.com/bulutklinik/{file_name}', file)     # send the file
        file.close()                                    # close file and FTP
        self.session.quit()
#png_file = r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\tahlil.png"
#excel_out = r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\Enabiz-Tahlilleri.xlsx"

#image = Image.open(png_file)

# Use pytesseract to perform OCR on the image
#text = pytesseract.image_to_string(image)
#print(text)
# Print the extracted text
if __name__ == "__main__":
    Sultan1Murat = Hastane()
    #Sultan1Murat.send_tahlil()
    veri = open(r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\Sultan1Murat.json",'rb')  
    Sultan1Murat.send_data(veri,"Sultan1Murat.json")
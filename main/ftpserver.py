import ftplib
import tabula
import tablib

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
pdf_file = r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\Enabiz-Tahlilleri.pdf"
excel_out = r"C:\Users\dest4\Desktop\bulutklinik-V1.2\main\server_data\Enabiz-Tahlilleri.xlsx"
tabula.convert_into(pdf_file, excel_out, output_format="csv", stream=True)
if __name__ == "__main__":
    Sultan1Murat = Hastane()
    Sultan1Murat.send_tahlil()
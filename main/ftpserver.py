import ftplib
session = ftplib.FTP('mt-ares.guzelhosting.com','morvecom','226Iry4Tkz')
file = open('pnomenia_predict/IM-0001-0001.jpeg','rb')                  # file to send
session.storbinary('STOR /matiricie.com/bulutklinik/image.jpg', file)     # send the file
file.close()                                    # close file and FTP
session.quit()
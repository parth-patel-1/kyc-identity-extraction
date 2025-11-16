import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr

image_file_name = 'C:/Users/pytho/Downloads/adi.jpg'

# Read the image
img = cv2.imread(image_file_name)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use the QR code detector to locate the QR code
qcd = cv2.QRCodeDetector()
retval, points, straight_qrcode = qcd.detectAndDecode(gray)

# Draw a bounding box around the detected QR code
if points is not None:
    x, y, w, h = cv2.boundingRect(points)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save the image with the bounding box
cv2.imwrite('image_with_bbox.png', img)

# Display the image
cv2.imshow('Image with Bounding Box', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the QR code data
if retval != '':
    qrData = retval
    print("QR Code Data:", qrData)

    if isSecureQr(qrData):
        print("This is a Secure Aadhaar QR Code")
    else:
        print("This is an Old Aadhaar QR Code")
else:
    print("No QR Code Detected")

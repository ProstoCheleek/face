import cv2

people = cv2.imread('IMG/1.jpg')
people2 = cv2.cvtColor(people, cv2.COLOR_BGR2RGB)

Faces = cv2.CascadeClassifier('body.xml')
results = Faces.detectMultiScale(people2, scaleFactor=10.1, minNeighbors=1)

print(results)

for (x,y,w,h) in results:
    cv2.rectangle(people, (x,y), (x+w, y+h), (0,0,255), thickness=3)

cv2.imshow('Face', people)
cv2.waitKey(0)
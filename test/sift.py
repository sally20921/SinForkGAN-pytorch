import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./dark-zurich.png')
img2 = cv2.imread('./dark-zurich-forkgan.png')
img3 = cv2.imread('./dark-zurich-day.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

key1, desc1 = sift.detectAndCompute(img1, None)
key2, desc2 = sift.detectAndCompute(img2, None)
key3, desc3 = sift.detectAndCompute(img3, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


daynight = bf.match(desc1, desc3)
daynight = sorted(daynight, key = lambda x: x.distance)

# filter matches using the Lowe's ratio test
ratio_threshold = 0.7
good_matches = []
for m,n in daynight:
  if m.distance < ratio_thresh * n.distance:
    good_matches.append(m)

#result1 = cv2.drawMatches(img1, key1, img3, key3, daynight, img3, flags=2)
resutl1 = cv2.drawMatches(img1, key1, img3, key3, good_matches, img3, flags=2)
plt.imsave('./sift-gray-night.png', result1, format='png')

#dayforkgan = bf.match(desc2, desc3)
#dayforkgan = sorted(dayforkgan, key= lambda x: x.distance)
#result2 = cv2.drawMatches(img2, key2, img3, key3, dayforkgan, img3, flags=2)
#plt.imsave('./sift-gray-forkgan.png', result2, format='png')

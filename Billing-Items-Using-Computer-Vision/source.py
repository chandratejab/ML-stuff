import numpy as np
import cv2
from label_images_custom import predictor

ITEMS_COST = {
	'Savlon Handwash': 100,
	'Otrivin Spray': 200,
	'Thread Vunda': 10,
	'Sunfeast Biscuit': 20,
	'Colin': 40,
	'Vaseline Liquid': 30
}

def startBilling(image_path):
	im = cv2.imread(image_path)
	im = cv2.resize(im, (4160, 2340))
	im_copy = cv2.imread(image_path)
	im_copy = cv2.resize(im_copy, (4160, 2340))
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,115,255,cv2.THRESH_BINARY)

	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Drawing rectangles into the color copy
	for cn in contours:
		x,y,wa,ha = cv2.boundingRect(cn)
		if wa<100 and ha<100:
			rect = cv2.minAreaRect(cn)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(im_copy, [box], -1, (0, 255, 0), -1)

	# Now converting the image into single channel as findContours doesnt support multiple channel images
	im_copy = cv2.cvtColor(im_copy,cv2.COLOR_BGR2GRAY)
	ret,im_copy = cv2.threshold(im_copy,70,255,cv2.THRESH_BINARY)
	im2, contours, hierarchy = cv2.findContours(im_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Finally taking out all the rectangles as objects and placing into dir
	idx=0
	all_items = []
	for cn in contours:
		x,y,wa,ha = cv2.boundingRect(cn)
		if wa>173 and ha>173:
			idx+=1
			new_img=im[y:y+ha,x:x+wa]
			cv2.imwrite('detected_items/'+str(idx) + '.png', new_img)
			all_items.append('detected_items/'+str(idx) + '.png')
			
	# cv2.imshow("thresh", thresh)
	# cv2.imshow("im", im)
	# cv2.imshow("im_copy", im_copy)
	# cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
	imS = cv2.resize(im_copy, (960, 540))                    # Resize image
	
	cv2.imshow("output", imS) 
	im = cv2.resize(im, (960, 540))     
	cv2.imshow("output2", im)                            # Show image
	cv2.waitKey(0)  
	cv2.destroyAllWindows()

	return all_items


items = startBilling('all_items.jpeg')

response = predictor(items,'labels.txt','retrained_graph.pb')
print(response)
# for item in items:
# 	# import pdb;pdb.set_trace()
# 	print(predictor(item,'labels.txt','retrained_graph.pb'))

print(items)
sum=0
for item in response:
	print("{}{}{}".format(item,' '*(30-len(item)),ITEMS_COST[item]))
	sum+=ITEMS_COST[item]

print("-"*40)
print("Total {}{}".format(' '*24,sum))
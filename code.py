# : Load sphere images and find the brightest spot. 
#Then calibrate the illumination direction for each image.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm


filename =["/gray.txt","/buddha.txt","/cat.txt","/owl.txt","/chrome.txt","/horse.txt","/rock.txt"] 
path="psmImages"
allPicture=[] 

for i in range(len(filename)): 
    with open(path+filename[i]) as f: 
        num = int(f.readline()) 
        temp =[] 
        for i in range(num+1): 
            pictureName =f.readline()[:-1]
            tgaImg=Image.open(pictureName) 
            temp.append(np.array(tgaImg)/255) 
        allPicture.append(temp) 

light_dir = [] 
with open(path+"/lights.txt") as f: 
    num = int(f.readline()) 
    for i in range(num): 
        dirVector =f.readline().split() 
        dirVector =list(map(np.float32,dirVector)) 
        light_dir.append(dirVector) 
L=np.array(light_dir)


# recover the normal map of the object
def getNormalMap(images , light_dir):
	all_b ,all_g ,all_r =[],[],[]
	for img in images[:-1]:
		b , g ,r = img[: ,: ,0] , img[: ,:,1] ,img[: ,: ,2]
		all_b.append(b.reshape((-1 ,)))
		all_g.append(g.reshape((-1, )))
		all_r.append(r.reshape((-1 ,)))
	all_b = np.array(all_b)
	all_g = np.array(all_g)
	all_r = np.array(all_r)

	x1 = np.dot(np.linalg.pinv(light_dir) , all_b)
	x2 = np.dot(np.linalg.pinv(light_dir) , all_g)
	x3 = np.dot(np.linalg.pinv(light_dir) , all_r)
	x = (x1 + x2 + x3)/3
	return x
all_Normal = []

for i in range(len(allPicture)):
	normalMatrix = getNormalMap(allPicture[i],L)
	all_Normal.append(normalMatrix)

#save normal map as a RGB image and show it
def transformNormal(mask , x , h ,w):
    for p in range(x.shape[1]):
        if mask_b[p]!=0:
            q = np.sqrt(np.sum(x[:,p]*x[: ,p]))

    img_normal = np.zeros((h*w , 3),dtype = np.float32)
    img_normal[: ,0] = x[0 ,:]
    img_normal[: ,1] = x[1 ,:]
    img_normal[: ,2] = x[2 ,:]
    img_normal = img_normal.reshape((h , w ,3))
    result = np.zeros((h , w,3) , dtype = np.uint8)
    result = np.uint8((img_normal+1)/2*255)
    return result

NormalRGB = []
for i in range(len(allPicture)):
    mask_b = allPicture[i][-1][:,:,0].reshape((-1,))
    images = allPicture[i]
    h = images[0].shape[0]
    w = images[0].shape[1]
    result =transformNormal(mask_b ,all_Normal[i] ,h ,w)
    NormalRGB.append(result)

plt.figure(figsize=(10 ,60 ) ,dpi=80)
for i in range(7):
    plt.subplot(7 ,1 ,i+1)
    plt.imshow(NormalRGB[i])

#calculate albedo
def getChannelAlbedo(I,channel_n,L):
    J=np.dot(L,channel_n) 
    JJ = J*J
    IJ=I*J
    
    np.seterr(divide='ignore', invalid='ignore')
    Kd=np.sum(IJ,0)/np.sum(JJ,0) 
    return Kd 


def getAlbedo(images,L): 
	all_b,all_g,all_r=[],[],[] 
	for img in images[:-1]: 
		img=img*images[-1] 
		b,g,r=img[:,:,0],img[:,:,1] ,img[:,:,2] 
		all_b.append(b.reshape((-1,))) 
		all_g.append(g.reshape((-1,)))
		all_r.append(r.reshape((-1,))) 

	all_b=np.array(all_b) 
	all_g=np.array(all_g) 
	all_r=np.array(all_r) 

	""""
	n_b =getPixelNormal(all_b,L)
	n_g =getPixelNormal(all_g,L)
	n_r =getPixelNormal(all_r,L)
	"""
	n_b=np.dot(np.linalg.pinv(L),all_b) 
	n_g=np.dot(np.linalg.pinv(L),all_g) 
	n_r=np.dot(np.linalg.pinv(L),all_r) 

	Albedo_b=getChannelAlbedo(all_b,n_b,L) 
	Albedo_g=getChannelAlbedo(all_g,n_g,L) 
	Albedo_r=getChannelAlbedo(all_r,n_r,L) 
	return [Albedo_b,Albedo_g,Albedo_r] 

allAlbedo = [] 
for i in range(len(allPicture)): 
	images=allPicture[i] 
	h=images[0].shape[0] 
	w=images[0].shape[1] 
	img_albedo=getAlbedo(images,L) 
	result=np.zeros((h,w,3))
	result[:,:,0]=img_albedo[0].reshape(h,w) 
	result[:,:,1] =img_albedo[1].reshape(h,w) 
	result[:,:,2]=img_albedo[2].reshape(h,w)
	allAlbedo.append(result) 

plt.figure(figsize=(9,20),dpi=80) 
for i in range(7): 
	plt.subplot(7,3,i*3+1) 
	plt.title("blue channel") 
	plt.imshow(allAlbedo[i][:,:,0],cmap='gray') 
	plt.subplot(7,3,i*3+2) 
	plt.title("green channel") 
	plt.imshow(allAlbedo[i][:,:,1],cmap='gray') 
	plt.subplot(7,3,i*3+3) 
	plt.title("red channel") 
	plt.imshow(allAlbedo[i][:,:,2],cmap='gray') 

# solve for the z values for each pixel.
import scipy 
import scipy.sparse.linalg 
def getDepthImage(mask,Normal):
	h,w=mask.shape
	obj_h,obj_w=np.where(mask!=0) 
	pixelNum=np.size(obj_h) 
	indexs=np.zeros((h,w)) 
	for i in range(pixelNum): 
		indexs[obj_h[i],obj_w[i]]=i 
		
	M=scipy.sparse.lil_matrix((2*pixelNum,pixelNum)) 
	v=np.zeros((2*pixelNum,1))
	for i in range(pixelNum): 
		x=obj_h[i] 
		y=obj_w[i] 
		n_y=N[x,y,0] 
		n_x=N[x,y,1] 
		n_z=N[x,y,2] 

		#z(x+1y)-z(x，y)=-nx/nz z(x，y+1) z(x，y)=-n 
		#(2*pixelNum,pixelNum)(10，-1)，v(2*pixelNum,1) 
		now_row=i*2 
		#those that  have (x，y+1)
		if mask[x,y+1]: 
			y1=indexs[x,y+1] 
			M[now_row,i]=-n_z 
			M[now_row,y1]=n_z
			v[now_row] =-n_y 
		# or (x,y-1)
		elif mask[x,y-1]: 
			y2=indexs[x,y-1] 
			M[now_row,i]=n_z
			M[now_row,y2]=-n_z 
			v[now_row]=-n_y 
		#or (x+1，y)
		now_row = i*2+1  
		if mask[x+1,y]: 
			x1=indexs[x+1,y] 
			M[now_row,i]=n_z 
			M[now_row,x1]=-n_z 
			v[now_row]=-n_x
		# or (x-1,y)
		elif mask[x-1,y]: 
			x2= indexs[x-1,y] 
			M[now_row,i]=-n_z 
			M[now_row,x2]=n_z
			v[now_row]=-n_z
	MTM = M.T @ M
	MTv = M.T @ v 
	Z=scipy.sparse.linalg.cg(MTM,MTv) 
	Z=Z[0] 
	std_z=np.std(Z,ddof=1) 
	mean_z=np.mean(Z) 
	z_zscore=(Z-mean_z)/std_z 

	outlier_ind =np.abs(z_zscore)>10 
	z_min=np.min(Z[~outlier_ind]) 
	z_max=np.max(Z[~outlier_ind]) 

	p=mask.astype('float') 
	for idx in range(pixelNum):
		h=obj_h[idx] 
		w=obj_w[idx] 
		p[h,w]=(Z[idx]-z_min)/(z_max - z_min)*255

	return p

all_Normal =[] 
for i in range(len(allPicture)):
	normalMatrix=getNormalMap(allPicture[i],L) 
	all_Normal.append(normalMatrix) 

allDepthImages=[] 
for i in range(len(all_Normal)): 
	mask=allPicture[i][-1][: ,:, 0] 
	img_normal=all_Normal[i] 
	N=np.zeros((h*w,3),dtype=np.float32) 
	N[:,0]=img_normal[0,:] 
	N[:,1]=img_normal[1,:] 
	N[:,2]=img_normal[2,:] 
	N=N.reshape((h,w,3)) 
	depthImage=getDepthImage(mask,N)
	allDepthImages.append(depthImage) 

plt.figure(figsize=(10,60),dpi=80) 
for i in range(len(all_Normal)): 
	plt.subplot(7,1,i+1) 
	plt.title(filename[i][1:-4]+"'s depth Image") 
	plt.imshow(allDepthImages[i],cmap='gray') 

#plot the 3D surface
 
for i in range(len(all_Normal)):
    Z = allDepthImages[i] 
    _3dh,_3dw=Z.shape[0],Z.shape[1]
    y=np.arange(0,_3dh,1) 
    x=np.arange(0,_3dw,1)
    X,Y=np.meshgrid(x,y) 
    fig=plt.figure() 
    ax=fig.gca(projection='3d')

    # Plot the surface. 
    surf=ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)
   
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




# Photometric Stereo

Most of the content in the notebook is borrowed from [Prof Steve Seitz](https://www.smseitz.com/) who developed in his Computer Vision class. Here is a [blog](https://www.cnblogs.com/linzzz98/articles/13622026.html) from a MSC student in our lab.

![](https://ai-studio-static-online.cdn.bcebos.com/93cd2cd7f34f4c0d9853ba946de781d6970e06715f904dad86a7f47c39860e0b)

In this project, you will finish a 3D computer vision task, Photometric Stereso, which constructs a height field from a series of images of a Lambertian object under different illuminaiton directions. Your code will be able to calibrate the lighting directions, find the best fit normal and albedo at each pixel, then find a surface which best matches the solved normals.

## 1. Calibration Lighting Directions

Before we calculate normals from images, we have to calibrate our capture setup. This includes determining the lighting intensity and direction, as well as the camera response function. For this project, we provide you with the dataset which have linearized camera response function, so you can treat pixel values as intensities. Second, all the light sources all appear to have the same brightness. You'll be solving for albedos relative to this brightness, which you can just assume is 1 in some arbitrary units. In other words, you don't need to think too much about the intensity of the light sources.

The one remaining calibration step you are going to do is calibrating lighting directions. One method of determining the direction of point light sources is to photograph a shiny chrome sphere in the same location as all the other objects. Since we know the shape of this object, we can determine the normal at any given point on its surface, and therefore we can also compute the reflection direction for the brightest spot on the surface.

    Write your code below to calculate lighting directions.

```
# TODO: Load sphere images and find the brightest spot. 
#Then calibrate the illumination direction for each image.
#unzip data/data120404/psmImages.zip 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor
from sklearn.preprocessing import normalize 

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



```

## 2. Calcualting Normals from Images

The appearance of diffuse objects can be modeled as $I=k_d\mathbf{n}^\mathrm{T}\mathbf{L}$ where $I$ is the pixel intensity, $k_d$ is the albedo, and $\mathbf{L}$ is the lighting direction (a unit vector), and $\mathbf{n}$ is the unit surface normal. (Since our images are already balanced as described above, we can assume the incoming radiance from each light is 1.) Assuming a single color channel, we can rewrite this as $I=(k_d\mathbf{n}^\mathrm{T})\mathbf{L}$, so the unknowns are together. With three or more different image samples under different lighting, we can solve for the product $\mathbf{g}=k_d\mathbf{n}$ by solving a linear least squares problem. The objective function is:

$$Q=\Sigma_i(I_i-\mathbf{g}^\mathrm{T}\mathbf{L_i})^2$$

To help deal with shadows and noise in dark pixels, its helpful to weight the solution by the pixel intensity: in other words, multiply by $I_i$:

$$Q=\Sigma_i(I_i^2-I_i\mathbf{g}^\mathrm{T}\mathbf{L_i})^2$$

The objective $Q$ is then minimized with respect to $\mathbf{g}$. Once we have the vector $\mathbf{g}=k_d\mathbf{n}$, the length of the vector is $k_d$ and the normalized direction gives $\mathbf{n}$.

Weighting each term by the image intensity reduces the influence of shadowed regions; however, it has the drawback of overweighting saturated pixels, due to specular highlights, for example. You can use the same weighting scheme we used in the HDRI project to address this issue.

```
# TODO: recover the normal map of the object
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
```

Save your normal map as a RGB image. Note that the 3 components of the normal is between [-1, 1], you need to scale it by (n+1)/2*255.

```
# TODO: save normal map as a RGB image and show it
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
```

## 3. Solving for color albedo

This gives a way to get the normal and albedo for one color channel. Once we have a normal $\mathbf{n}$ for each pixel, we can solve for the albedos by another least squares solution. The objective function is:

$$ Q = \Sigma_i{(I_i - k_d \mathbf{n}^\mathrm{T}\mathbf{L_i})^2} $$

To minimize it, differentiate with respect to $k_d$, and set to zero:

$$ \frac{\partial Q}{\partial k_d} = \Sigma_i{2(I_i - k_d \mathbf{n}^\mathrm{T}\mathbf{L_i})(\mathbf{n}^\mathrm{T}\mathbf{L_i})} = 0 $$

$$  k_d = \frac{\Sigma_i{I_i\mathbf{n}^\mathrm{T}\mathbf{L_i}}}{\Sigma_i{(\mathbf{n}^\mathrm{T}\mathbf{L_i})^2}} $$


Writing $J_i = \mathbf{L_i} \cdot \mathbf{n}$, we can also write this more concisely as a ratio of dot products: $k_d = \frac{\mathbf{I} \cdot \mathbf{J} } {\mathbf{J} \cdot \mathbf{J} }$. This can be done for each channel independently to obtain a per-channel albedo.

```
# TODO: calculate albedo
def getChannelAlbedo(I,channel_n,L):
    J=np.dot(L,channel_n) 
    JJ = J*J
    IJ=I*J

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
	al_r=np.array(all_r) 

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

	
```

## 4. Least-squares surface fitting
Next we'll have to find a surface which has these normals, assuming such a surface exists. We will again use a least-squares technique to find the surface that best fits these normals. Here's one way of posing this problem as a least squares optimization.

If the normals are perpendicular to the surface, then they'll be perpendicular to any vector on the surface. We can construct vectors on the surface using the edges that will be formed by neighbouring pixels in the height map. Consider a pixel (i,j) and its neighbour to the right. They will have an edge with direction:

(i+1, j, z(i+1,j)) - (i, j, z(i,j)) = (1, 0, z(i+1,j) - z(i,j))

This vector is perpendicular to the normal n, which means its dot product with n will be zero:

(1, 0, z(i+1,j) - z(i,j)) . n = 0

$$ n_x + n_z(z(i+1,j) - z(i,j) = 0 $$

Similarly, in the vertical direction:

$$ n_y + n_z(z(i,j+1) - z(i,j) = 0 $$

We can construct similar constraints for all of the pixels which have neighbours, which gives us roughly twice as many constraints as unknowns (the z values). These can be written as the matrix equation Mz = v. The least squares solution solves the equation $\mathbf{M}^\mathrm{T}\mathbf{M}\mathbf{z}=\mathbf{M}^\mathrm{T}\mathbf{v}$. However, the matrix $\mathbf{M}^\mathrm{T}\mathbf{M}$ will still be really really big! It will have as many rows and columns as their are pixels in your image. Even for a small image of 100x100 pixels, the matrix will have 10^8 entries!

Fortunately, most of the entries are zero, and there are some clever ways of representing such matrices and solving linear systems with them. 

(a) most of the entries are zero, and there are some clever ways of representing such matrices called sparse matrices. You can figure out where those non-zero values are, put them in a sparse matrix, and then solve the linear system.

(b) there are iterative algorithms that allow you to solve linear system without explictly storing the matrix in memory, such as the conjugate gradient method:

import scipy
scipy.sparse.linalg.cg(A, b)

(c) use SVD.

```
# TODO: solve for the z values for each pixel.
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


```

Plot the reconstructed 3D surface using 3D plot tool of matplotlib, see more information [here](https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html).

```
# TODo: plot the 3D surface
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

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

```


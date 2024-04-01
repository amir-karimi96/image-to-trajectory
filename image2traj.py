import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np

import cv2 
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from skimage import morphology
from skimage import transform as tf
import pickle

to_pil = transforms.ToPILImage()



# Define a CNN model
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2*12+2*12)  # Output layer with vector size 12 + variance

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        # x = torch.tanh(self.conv1(x))
        # x = torch.tanh(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x

def create_circle_kernel(radius):
    diameter = 2 * radius + 1
    y_indices, x_indices = np.indices((diameter, diameter))
    center = radius, radius
    distance_from_center = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)
    circle_kernel = distance_from_center <= radius
    return circle_kernel.astype(np.uint8)

def rbf(centers, width):   
    b = lambda x: np.stack([np.exp(-(x - c_i) ** 2 / (2 * h_i)) for c_i, h_i in zip(centers, width)]).T  # eq 7
    return lambda x: b(x) / np.sum(b(x), axis=1, keepdims=True)  # eq 8

def linear_rbf(A, t, duration):
        
        n_features = len(A)

        params = A.reshape(1, n_features)
        
        h = (1. + 1 / n_features) / n_features 
        h = h ** 2 * 0.5

        bandwidths = np.repeat([h], n_features, axis=0)
        centers = np.linspace(0 - 0.5 / n_features, 1 + 0.5 / n_features, n_features)
        
        phi = rbf(centers, bandwidths)(t/duration)
        return np.matmul(phi, params.T)

def rbf2traj(w,N):
    
    t = np.linspace(0.0,1.0,N)
    y = linear_rbf(w,t,1.0)
    return y

def rbf2image(wx,wy):
    w=28
    h=28
    image = np.zeros((w,h))
    X = np.clip((w/2*rbf2traj(wx, 100)+w/2).astype(np.int32), 0, w-1)
    Y = np.clip((w/2*rbf2traj(wy, 100)+w/2).astype(np.int32),0,h-1)

    image[(X,Y)] = 1.0
    # Define a kernel for dilation
    kernel = np.ones((2, 2), np.uint8)

    # Dilate the image
    image = cv2.dilate(image, kernel, iterations=1)
    return image

kernel = create_circle_kernel(10)
def tr2image(tx,ty):
    w=280
    h=280
    image = np.zeros((w,h))
    X = (w/2*tx+w/2).astype(np.int32).reshape(-1)
    Y = (w/2*ty+w/2).astype(np.int32).reshape(-1)

    image[(X,Y)] = 1.0
    for x,y in zip(X,Y):
        # print(x,y)
        image[np.int32(x-10):np.int32(x+11),np.int32(y-10):np.int32(y+11)] = (image[np.int32(x-10):np.int32(x+11),np.int32(y-10):np.int32(y+11)]+kernel)>0
    # Define a kernel for dilation
    # kernel = create_circle_kernel(10)

    # Dilate the image
    # image = cv2.dilate(image, kernel, iterations=1)
    return image

def inv_rbf(x,n):
    n_features=n
    h = (1. + 1 / n_features) / n_features 
    h = h ** 2 * 0.5

    bandwidths = np.repeat([h], n_features, axis=0)
    centers = np.linspace(0 - 0.5 / n_features, 1 + 0.5 / n_features, n_features)
    if len(x) == 1:
        w= np.ones((n,))*x[0]
        return w.reshape(-1)
    if len(x) < n_features:
        xx = []
        for j in range(len(x)-1):
            xx.extend(np.linspace(x[j],x[j+1],n_features, endpoint=False))
        x = np.array(xx)
    t = np.linspace(0.0,1.0,len(x))
    
    phi = rbf(centers, bandwidths)(t)#t/duration])
    PHI_A = np.matmul(phi.T,phi)#+1e-6*np.eye(3)
    PHI_B = np.matmul(x.reshape(1,-1), phi)
    # print(PHI_B)
    return np.matmul( np.linalg.inv(PHI_A), PHI_B.T).reshape(-1)

def simulate_movement(Ax, Ay,num_steps):
    # Initial position and velocity
    x = np.random.random()-0.25
    y = np.random.random()-0.25
    velocity_x = 0
    velocity_y = 0
    acceleration_x = 0
    acceleration_y = 0
    dt = 1/num_steps
    # Initialize lists to store trajectory
    trajectory_x = [x]
    trajectory_y = [y]

    # Simulate movement
    for i in range(num_steps):
        # Update acceleration
        dacceleration_x = np.random.uniform(-.01, .0100)
        dacceleration_y = np.random.uniform(-.0100, .0100)

        acceleration_x = Ax[i]#dacceleration_x*dt
        acceleration_y = Ay[i]#dacceleration_y*dt
        
        # Update velocity
        velocity_x += acceleration_x*dt
        velocity_y += acceleration_y*dt

        velocity_x = Ax[i]
        velocity_y = Ay[i]

        # Update position
        x += velocity_x*dt
        y += velocity_y*dt

        # Append current position to trajectory
        trajectory_x.append(x)
        trajectory_y.append(y)

    return trajectory_x, trajectory_y



# dataset generator 1
class CustomDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = []  # List to store generated data
        
        
        while len(self.data) < size:  
            num_steps = 200

            # random parameters w for trajectory x
            w = 2*np.random.random((6,))-1.0
            tx = rbf2traj(w, num_steps).reshape(-1)
                                    
            # random parameters w for trajectory x
            w = 2*np.random.random((6,))-1.0
            ty = rbf2traj(w, num_steps).reshape(-1)
            
            mx = np.max(np.abs(tx))
            my = np.max(np.abs(ty))
            if mx > 0.9:
                tx = tx / mx*0.9
            elif my > 0.9:
                ty=ty/my*0.9
            goal_traj_x = np.clip(tx,-0.9,0.9)
            goal_traj_y = np.clip(ty,-0.9,0.9)
            xx = [goal_traj_x[0]]
            yy = [goal_traj_y[0]]

            # sample equally spaced points in 2d
            for i in range(num_steps):
                d = ((goal_traj_x[i]-xx[-1])**2 + (goal_traj_y[i]-yy[-1])**2)**0.5
                if d >= 1*1/28:
                    xx.append(goal_traj_x[i])
                    yy.append(goal_traj_y[i])

            goal_traj_x = np.array(xx)
            goal_traj_y = np.array(yy)
            
            wx = inv_rbf(goal_traj_x , 6)
            wy = inv_rbf(goal_traj_y , 6)

            # cv2.imshow('a',np.uint8(255*tr2image(rbf2traj(wx,100),rbf2traj(wy,100))))
            # cv2.waitKey()
            # goal_traj_x = rbf2traj(wx,100)
            # goal_traj_y = rbf2traj(wy,100)
            # goal_traj_x = np.clip(goal_traj_x,-0.9,0.9)
            # goal_traj_y = np.clip(goal_traj_y,-0.9,0.9)

            #generate image based on vector
            input_image = tr2image(goal_traj_x, goal_traj_y)
            # cv2.imshow('a',np.uint8(255*input_image))
            # cv2.waitKey()

            if np.sum(input_image) / len(goal_traj_x) > 115: # check if it is self-colliding too much 
                
                input_image0 = cv2.resize(input_image, (28, 28), interpolation=cv2.INTER_AREA)
                
                # cv2.imshow('a',np.uint8(255*rbf2image(wx[::-1],wy[::-1])))
                # cv2.waitKey()

                # add multiple data using symmetry 

                self.add_data_custom(input_image0, wx, wy)
                self.add_data_custom(input_image0, wx[::-1], wy[::-1])

                # x -> -x
                self.add_data_custom(cv2.flip(input_image0,0), -wx, wy)
                self.add_data_custom(cv2.flip(input_image0,0), -wx[::-1], wy[::-1])

                # y -> -y
                self.add_data_custom(cv2.flip(input_image0,1), wx, -wy)
                self.add_data_custom(cv2.flip(input_image0,1), wx[::-1], -wy[::-1])
                
                # x -> -x , y -> -y
                self.add_data_custom(cv2.flip(cv2.flip(input_image0,0), 1), -wx, -wy)
                self.add_data_custom(cv2.flip(cv2.flip(input_image0,0), 1), -wx[::-1], -wy[::-1])

    def add_data_custom(self,input_image, p1,p2):
        input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)

        output_vector = torch.tensor(np.concatenate((p1,p2)), dtype=torch.float32)
        output_vector += 0.01*torch.randn_like(output_vector)
        self.data.append((input_image, output_vector))
                

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

# dataset generator for random digits
class RandomDigitsDataset(Dataset):
    def __init__(self, size):
        self.size = 2*size
        self.data = []  # List to store generated data
        y_5 = [186, 176, 166, 156, 146, 136, 126, 116, 106, 101, 102, 102, 103, 103, 103, 103, 113, 123, 133, 143, 153, 163, 173, 181, 189, 195, 198, 198, 198, 194, 188, 180, 172, 162, 152, 142, 132, 123, 114] 
        x_5 = [75, 75, 75, 75, 74, 74, 73, 73, 73, 83, 93, 103, 113, 123, 133, 143, 143, 140, 138, 137, 136, 136, 138, 144, 150, 158, 168, 178, 188, 198, 206, 214, 221, 225, 226, 226, 222, 217, 212]
        y_4 = [101, 96, 93, 91, 88, 84, 81, 78, 88, 98, 108, 119, 129, 139, 149, 159, 169, 179, 189, 199, 193, 186, 179, 173, 167, 164, 163, 163, 163, 163, 163, 163, 163, 163, 163, 162, 162, 162, 162] 
        x_4 = [70, 81, 91, 102, 112, 122, 132, 142, 144, 144, 144, 145, 146, 146, 146, 146, 146, 146, 147, 147, 138, 130, 121, 113, 105, 95, 106, 117, 128, 139, 149, 160, 170, 180, 190, 201, 211, 221, 231]
        y_2 = [75, 76, 81, 87, 95, 104, 113, 123, 133, 143, 153, 163, 173, 181, 187, 192, 195, 196, 195, 190, 183, 176, 168, 160, 152, 144, 136, 128, 119, 110, 102, 93, 86, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196] 
        x_2 = [107, 97, 88, 80, 74, 69, 64, 62, 60, 59, 59, 62, 66, 72, 80, 89, 99, 109, 119, 128, 136, 144, 150, 157, 163, 169, 175, 181, 186, 191, 197, 203, 211, 221, 225, 226, 226, 224, 222, 221, 221, 220, 220, 220, 220]
        y_1 = [141, 139, 140, 141, 141, 141, 141, 141, 142, 142, 143, 143, 143, 144, 144]
        x_1 = [57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197]
        y_3 = [95, 107, 117, 129, 141, 153, 165, 170, 173, 173, 169, 159, 149, 140, 128, 142, 152, 163, 172, 180, 185, 189, 190, 186, 180, 170, 158, 145, 132, 120, 112] 
        x_3 = [71, 65, 62, 59, 58, 59, 68, 80, 91, 102, 113, 124, 131, 136, 138, 137, 137, 141, 146, 155, 168, 179, 191, 203, 211, 217, 223, 225, 225, 219, 211]
        y_6 = [167, 157, 147, 139, 132, 124, 118, 112, 105, 102, 100, 99, 97, 97, 97, 98, 102, 107, 113, 121, 132, 144, 155, 165, 171, 177, 179, 178, 171, 166, 158, 148, 138, 127, 118, 110, 104] 
        x_6 = [62, 62, 65, 71, 79, 86, 94, 103, 112, 122, 133, 143, 153, 163, 173, 184, 195, 205, 213, 219, 220, 220, 219, 214, 206, 198, 188, 177, 169, 159, 152, 149, 149, 150, 156, 162, 171]
        y_7 = [ 79, 90, 102, 112, 123, 133, 143, 154, 164, 174, 185, 195, 196, 189, 184, 177, 171, 164, 157, 150, 145, 139, 133, 127, 122] 
        x_7 = [85, 86, 88, 88, 90, 90, 90, 88, 86, 84, 82, 81, 92, 102, 112, 126, 138, 150, 162, 173, 184, 195, 205, 214, 223]
        y_8 = [137, 127, 118, 110, 104, 101, 100, 100, 104, 112, 122, 133, 143, 153, 163, 174, 184, 190, 196, 199, 197, 190, 183, 174, 165, 156, 147, 139, 131, 123, 118, 112, 106, 103, 102, 106, 112, 120, 130, 140, 150, 160, 170, 180, 188, 194, 196, 195, 190, 184, 178, 170, 162, 154, 145] 
        x_8 = [136, 133, 127, 120, 112, 102, 92, 82, 72, 66, 62, 61, 60, 60, 61, 64, 68, 76, 84, 94, 104, 112, 120, 125, 130, 135, 140, 148, 155, 163, 172, 180, 189, 199, 209, 219, 227, 233, 235, 236, 236, 236, 236, 232, 226, 218, 208, 198, 188, 180, 172, 164, 157, 151, 145]
        y_9 = [166, 159, 149, 139, 129, 119, 110, 104, 103, 105, 107, 112, 120, 128, 138, 148, 159, 168, 175, 180, 185, 187, 188, 188, 188, 185, 183, 180, 176, 169, 161, 153, 143, 132] 
        x_9 = [137, 145, 148, 148, 148, 147, 142, 133, 123, 113, 103, 92, 85, 78, 74, 73, 73, 78, 86, 99, 112, 123, 133, 143, 153, 164, 174, 186, 197, 207, 215, 221, 227, 231]
        y_0 = [177, 170, 161, 152, 142, 132, 122, 112, 103, 96, 91, 87, 85, 84, 83, 83, 84, 87, 91, 99, 108, 117, 128, 138, 148, 158, 167, 173, 177, 181, 184, 185, 185, 185, 183, 179, 172, 164] 
        x_0 = [102, 94, 89, 83, 81, 79, 80, 85, 92, 100, 109, 119, 129, 139, 149, 160, 171, 183, 194, 201, 207, 212, 214, 214, 212, 208, 200, 189, 179, 169, 158, 146, 135, 124, 113, 103, 93, 87]
        xs = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9,x_0]
        ys = [y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9,y_0]
        wxs = []
        wys = []
        for i in range(10):
            wxs.append(inv_rbf(np.array(xs[i])/140 - 1, 6))
            wys.append(inv_rbf(np.array(ys[i])/140 - 1, 6))

        for _ in range(size):
            k = np.random.randint(0,10)
            
            wx = wxs[k]+rbf2traj(np.random.random((4,))*0.6-0.3, 6).reshape(-1)#(np.random.random((10,))*0.4-0.2)
            wy = wys[k]+rbf2traj(np.random.random((4,))*0.6-0.3, 6).reshape(-1)#(np.random.random((10,))*0.4-0.2)
            wx = wx * (1+0.4*np.random.random()-0.2)
            wy = wy * (1+0.4*np.random.random()-0.2)
            
            goal_traj_x = rbf2traj(wx, 200)
            goal_traj_y = rbf2traj(wy, 200)
            mx = np.max(np.abs(goal_traj_x))
            my = np.max(np.abs(goal_traj_y))
            if mx > 0.9:
                goal_traj_x = goal_traj_x / mx*0.9
                
            elif my > 0.9:
                
                goal_traj_y=goal_traj_y/my*0.9
            
            goal_traj_x = np.clip(goal_traj_x,-0.95,0.95)
            goal_traj_y = np.clip(goal_traj_y,-0.95,0.95)
            xx = [goal_traj_x[0]]
            yy = [goal_traj_y[0]]
            for i in range(200):
                d = ((goal_traj_x[i]-xx[-1])**2 + (goal_traj_y[i]-yy[-1])**2)**0.5
                if d >= 1*1/28:
                    xx.append(goal_traj_x[i])
                    yy.append(goal_traj_y[i])

            goal_traj_x = np.array(xx)
            goal_traj_y = np.array(yy)
            wx = inv_rbf(goal_traj_x , 6)
            wy = inv_rbf(goal_traj_y , 6)
            
            
            # generate image based on vector
            input_image = tr2image(goal_traj_x, goal_traj_y)
            input_image = cv2.resize(input_image, (28, 28), interpolation=cv2.INTER_AREA)
            # cv2.imshow('a',np.uint8(255*input_image))
            # cv2.waitKey()
        
            # Convert to tensor
            input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
            output_vector = torch.tensor(np.concatenate((wx,wy)), dtype=torch.float32)
            # print(input_image, output_vector)
            self.data.append((input_image, output_vector))
            self.data.append((input_image, torch.concatenate((torch.flip(output_vector[:6],[0]), torch.flip(output_vector[6:],[0])))))
            

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def preprocess(x):
    x = np.array(x)/255
    # x = np.where(x < 0.25, 0.0, 1.0)
    # x = cv2.resize(x, (280,280))# interpolation=cv2.INTER_LINEAR)
    # skeleton = morphology.skeletonize(x)#method='zhang')
    # # x = cv2.resize(x, (280,280), interpolation=cv2.INTER_LINEAR)
    # # x = cv2.resize(x, (28,28), interpolation=cv2.INTER_AREA)
    # resized_skeleton = tf.resize(skeleton, (280,280), anti_aliasing=False).astype(np.float32)
    # rr = np.zeros((280,280),dtype=np.float32)
    # rr= resized_skeleton
    # X,Y = np.where(rr>0)
    # for x,y in zip(X,Y):
    #     # print(x,y)
    #     try:
    #         rr[np.int32(x-10):np.int32(x+11),np.int32(y-10):np.int32(y+11)] = (rr[np.int32(x-10):np.int32(x+11),np.int32(y-10):np.int32(y+11)]+kernel)>0
    #     except:
    #         pass
    # x = cv2.resize(rr, (28,28), interpolation=cv2.INTER_AREA)
    return (255*x).astype(np.uint8)
    return torch.where(x < 0.35, torch.tensor(0.0), torch.tensor(1.0))


def CD(s,r):
    # print('p started', s)
    while True:
        dd = CustomDataset(s)
        r.put(dd)

# Save dataset
def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

# Load dataset
def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == '__main__':
    transform=transforms.Compose([
            transforms.Lambda(preprocess),
            transforms.ToTensor(),
            
            # transforms.Normalize((0.1307,), (0.3081,))
            ])
    

    # Prepare the dataset
    dataset_size = 10000  # Choose your desired dataset size
    batch_size = 1000  # Choose batch size

    test_dataset = RandomDigitsDataset(1000)
    
    # test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    dataset = CustomDataset(1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define the loss function and optimizer
    model = CNNRegression()
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    try:
        
        model.load_state_dict(torch.load('image_2_traj_model.pt'))
        
    except:
        import multiprocessing
        
        result_queue = multiprocessing.Queue(maxsize=4)
        process1 = multiprocessing.Process(target=CD, args=(1000, result_queue))
        process1.start()
        
        # Step 5: Write the training loop
        iteration = 1000
        for i in range(iteration):
            num_epochs = 4 
            # dataset = result_queue.get()
            # process.join()
            dataset = result_queue.get()
            # dataset = CustomDataset(1000)
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
            
            for epoch in range(num_epochs):
                
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    logvar1 = outputs[:,24:36]
                    logvar2 = outputs[:,36:]
                    var1 = torch.exp(logvar1)+1e-4
                    var2 = torch.exp(logvar2)+1e-4

                    flipped_outputs = torch.concatenate((torch.flip(outputs[:,:6],[1]), torch.flip(outputs[:,6:12],[1])), dim=1)
                    # flipped_targets = torch.concatenate((torch.flip(targets[:,:6],[1]), torch.flip(targets[:,6:],[1])), dim=1)
                    
                    lp1i = -0.5 * ((targets - outputs[:,:12]) ** 2) / var1 - 0.5*torch.log(2 * torch.pi * var1)
                    p1 = torch.exp(lp1i.sum(dim=1))

                    lp2i = -0.5 * ((targets - flipped_outputs) ** 2) / var2 - 0.5*torch.log(2 * torch.pi * var2)
                    p2 = torch.exp(lp2i.sum(dim=1))
                    
                    loglikelihoods = torch.logsumexp(torch.stack([lp1i.sum(dim=1), lp2i.sum(dim=1)], dim=1), dim=1)-0.7
                    
                    loss = -loglikelihoods.mean()

                    loss.backward()
                    optimizer.step()
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                
                if epoch % 5==0:
                    with torch.no_grad():
                        for inputs, targets in dataloader:
                            outputs = model(inputs)
                        output = outputs[0].numpy().reshape(-1)
                        output = output[:12]
                        im = inputs[0].reshape(28,28).numpy()
                        im = np.concatenate((im,rbf2image(output[:6],output[6:12]) ), axis=1)# Convert the torch tensor to a PIL Image
                        pil_img = torch.tensor(im,dtype=torch.float32)

                        # Save the PIL Image using torchvision's save_image function
                        save_image(pil_img, 'output_image.jpg')
                    torch.save(model.state_dict(),'image_2_traj_model.pt')

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            output = outputs[0].numpy().reshape(-1)
            output = output[:12]
            im = inputs[0].reshape(28,28).numpy()
            
            im = np.concatenate((im,rbf2image(output[:6],output[6:12]) ), axis=1)
            pil_img = torch.tensor(im,dtype=torch.float32)

            # Save the PIL Image using torchvision's save_image function
            save_image(pil_img, 'output_image.jpg')
            cv2.imshow('a',np.uint8(255*im))
            cv2.waitKey()
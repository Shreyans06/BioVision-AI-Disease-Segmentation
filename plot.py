import numpy as np 
import matplotlib.pyplot as plt 
import json

# set width of bar 
barWidth = 0.25
fig = plt.subplots( figsize =(16, 10)) 

f = open('./metrics/UNET_224x224_resnext101_32x8d_imagenet_run_10_val_metric.json')
data = json.load(f)

for k , v in data.items():
    param = list(v.keys())

metrics = dict(map(lambda i : (param[i] , []) , range(len(param))))

# print(data)
for k , v in data.items():
    for i in range(len(param)):
        metrics[param[i]] += [v[param[i]]] 

# Calculate the mean of each list
# means = np.mean(metrics.values(), axis=0)
print(metrics)
metrics.pop("loss")
metrics.pop("Dice")
print(metrics)
# Replace the list values with their mean
metrics = {key : np.mean(value) for key, value in metrics.items()}

print(metrics)

# set height of bar 
IoU = [] 
Dice = [] 

for k , v in metrics.items():
    IoU.append(round(v , 2))
    Dice.append( round(2 * v / (1 + v) , 2) )

# Set position of bar on X axis 
br1 = np.arange(len(IoU)) 
br2 = [x + barWidth for x in br1] 



# Make the plot
plt.bar(br1, IoU, color ='r', width = barWidth, 
            edgecolor ='grey', label ='IoU') 

plt.bar(br2, Dice, color ='b', width = barWidth, 
            edgecolor ='grey', label ='Dice') 
    
# Adding Xticks 
plt.xlabel('Class metrics', fontweight ='bold', fontsize = 15) 
plt.ylabel('Values', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(IoU))], 
		list(metrics.keys()) , rotation = 50)
plt.title("UNet model Class wise metrics")
plt.legend()
plt.savefig(f"./metrics/classIoU.png")
plt.show() 


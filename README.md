# DL_EXP1 Developing a Neural Network Regression Model
# Developed BY :P.PRAMISHA
# REG NO:212224230203
# AIM
To develop a neural network regression model for the given dataset.
# ALGORITHM

**Step 1:** Start the program.

**Step 2:** Import the required libraries such as PyTorch, Pandas, Scikit-learn, and Matplotlib.

**Step 3:** Load the regression dataset from the CSV file.

**Step 4:** Separate the dataset into input features (X) and target values (Y).

**Step 5:** Split the dataset into training and testing sets.

**Step 6:** Normalize the input features using the MinMaxScaler.

**Step 7:** Convert the training and testing data into PyTorch tensors.

**Step 8:** Define a neural network model with an input layer, hidden layers using ReLU activation, and an output layer.

**Step 9:** Initialize the loss function as Mean Squared Error (MSE) and the optimizer as Adam.

**Step 10:** Train the model by performing forward propagation, calculating the loss, applying backpropagation, and updating the weights for the specified number of epochs.

**Step 11:** Predict the output for new input data using the trained model.

**Step 12:** Evaluate the model using the test dataset and calculate the test loss.

**Step 13:** Plot the training loss against the epochs to visualize the learning process.

**Step 14:** Display the predicted output, test loss, and training loss graph.

**Step 15:** Stop the program.

# PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
```
```
df = pd.read_csv("C:\\Users\\varal\\Downloads\\Deep Learning\\Deep Learning\\Exp-1.csv")
df
```
<img width="356" height="542" alt="image" src="https://github.com/user-attachments/assets/cbbec432-3230-45da-b46f-d08788a4db0a" />

```
x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)
```

```
scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)
```

```
xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)
```
```
class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(), 
            nn.Linear(16,8), 
            nn.ReLU(), 
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)
```

```
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
```

```
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())
```
<img width="456" height="533" alt="image" src="https://github.com/user-attachments/assets/9113ff5f-e77e-4d8b-bcb0-46b26e09dab4" />

```
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())
```

<img width="355" height="67" alt="image" src="https://github.com/user-attachments/assets/2428fa92-2a82-4ca9-905e-e1087424cdea" />

```
with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)
```
<img width="341" height="82" alt="image" src="https://github.com/user-attachments/assets/a8e9f901-6066-45b4-ac4d-3a7b0c196efa" />

```
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```
<img width="915" height="587" alt="image" src="https://github.com/user-attachments/assets/86e9583b-4747-4430-9930-4526332553fb" />


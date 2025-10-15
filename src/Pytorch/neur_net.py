import torch, pandas as pd
df = pd.read_csv("cancer.csv")
X = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)
y = torch.tensor((df.iloc[:, 1] == 'M').astype(int).values, dtype=torch.long)
model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
opt = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(200): opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(X), y); loss.backward(); opt.step()
print((model(X).argmax(1) == y).float().mean().item())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

pems04_data = np.load('data/PEMS04/PEMS04.npz')

print(pems04_data.files)

print(pems04_data['data'].shape)



flow = pems04_data['data'][:,0,0]
# spead = pems04_data['data'][:,0,0]
# occupy = pems04_data['data'][:,0,0]
fig = plt.figure(figsize=(15, 5 ))
plt.title('traffic flow in San Francisco')
plt.xlabel('day')
plt.ylabel('traffic flow')
plt.plot(np.arange(len(flow), flow, lineStyles='-'))
fig.autofmt_xdate(rotation=45)
plt.show()
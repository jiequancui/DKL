import sys
import torch
import os

checkpoint = sys.argv[1]+'/ours-model-epoch{}.pt'
tau = float(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])

print(tau, " ******* tau ******", start, "************ start ******8")

new_state_dict={}
for i in range(1,end+1):
    print(checkpoint.format(i))
    state_dict = torch.load(checkpoint.format(i))
    #print(state_dict.keys(), " 9999999999999999999 ")
    if i == start:
        for key,value in state_dict.items():
            new_state_dict[key] = value
        print("start ************** ", i)
    elif i > start:
        for key,value in state_dict.items():
            new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]
        print(i, " momentum")

torch.save(new_state_dict,os.path.join(sys.argv[1], 'ours-model-epoch-SWA{}{}{}.pt'.format(tau,start,end)))


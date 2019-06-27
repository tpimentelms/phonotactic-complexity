import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rfolder = 'results/northeuralex/cv/'
rfolder_inventory = 'results/northeuralex/inventory/'
rfolder_artificial = 'results/northeuralex/artificial/%s/cv/orig/'
rfolder_artificial_harmony = rfolder_artificial % 'harmony'
rfolder_artificial_devoicing = rfolder_artificial % 'devoicing'

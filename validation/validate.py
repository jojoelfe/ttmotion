import pooch
import torch
import tifffile
import mrcfile
import einops

from ttmotion.estimate_motion import estimate_motion

gain = pooch.retrieve("https://ftp.ebi.ac.uk/empiar/world_availability/10491/data/gain_ref.mrc",known_hash=None)
movie1 = pooch.retrieve("https://ftp.ebi.ac.uk/empiar/world_availability/10491/data/frameseries/2Dvs3D_82-1_00001_-0.0_Aug05_08.40.22.tif",known_hash=None)

with mrcfile.open(gain) as f:
    gain = torch.tensor(f.data)
    # Flip in Y
    gain = torch.flip(gain, [0])
image = torch.tensor(tifffile.imread(movie1))
image = image * gain
image = (image - torch.mean(image)) / torch.std(image)
#mrcfile.write("gain_corrim.mrc", einops.reduce(image, 't h w -> h w', reduction='sum').numpy(), overwrite=True)

estimate_motion(
    image=image,
    deformation_field_resolution=(5, 1, 1),
)
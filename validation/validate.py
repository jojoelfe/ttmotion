import pooch
import torch
import tifffile
import mrcfile

from ttmotion.estimate_motion import estimate_motion

gain = pooch.retrieve("https://ftp.ebi.ac.uk/empiar/world_availability/10491/data/gain_ref.mrc",known_hash=None)
movie1 = pooch.retrieve("https://ftp.ebi.ac.uk/empiar/world_availability/10491/data/frameseries/2Dvs3D_82-1_00001_-0.0_Aug05_08.40.22.tif",known_hash=None)



with mrcfile.open(gain) as f:
    gain = torch.tensor(f.data)
image = torch.tensor(tifffile.imread(movie1))
# EMPIAR notes say gain is flipped in Y
image = image/gain
image = (image - torch.mean(image)) / torch.std(image)


estimate_motion(
    image=image,
    deformation_field_resolution=(5, 1, 1),
)
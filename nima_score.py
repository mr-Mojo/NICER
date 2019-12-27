from imports import *
from utils import get_tensor_mean_as_float
from neural_models import NIMA_VGG
from config import nima_checkpoint_path
# todo: load NIMA, load img, predict, plot img

device = 'cpu'

nima_checkpoint_path = r'C:\Users\Michi\Downloads\epoch-57.pkl'
nima = NIMA_VGG(models.vgg16(pretrained=True))
nima.load_state_dict(torch.load(nima_checkpoint_path, map_location=device))
nima.eval()
nima.to(device)

img_path = r'C:\Users\Michi\Downloads\nima_imgs\img3860_exp_neg1.jpg'
pil_img = Image.open(img_path)

nima_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor = nima_transform(pil_img).unsqueeze(dim=0)

result = nima(img_tensor)

score = get_tensor_mean_as_float(result)
print("Score: %f" % score)

plt.imshow(pil_img)
plt.show()



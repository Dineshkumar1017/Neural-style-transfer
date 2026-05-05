import torch
from PIL import Image
from torchvision import transforms
from utils.models import VGGEncoder, Decoder
from app import style_transfer, save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(r'C:\Users\DINESH KUMAR\Desktop\NST\experiment\final_exp\decoder_10.pth', map_location=device))
encoder.eval()
decoder.eval()

content = Image.open('examples/brad_pitt.jpg').convert('RGB')
style1 = Image.open('examples/sketch.png').convert('RGB')
style2 = Image.open('examples/picasso_seated_nude_hr.jpg').convert('RGB')

out1 = style_transfer(content, style1, encoder, decoder, 1.0, device)
save_image(out1, 'examples/brad_pitt_sketch_result.jpg')

out2 = style_transfer(content, style2, encoder, decoder, 1.0, device)
save_image(out2, 'examples/brad_pitt_picasso_result.jpg')

print("Images generated successfully!")

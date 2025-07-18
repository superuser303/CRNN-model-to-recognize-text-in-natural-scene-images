import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model.crnn import CRNN
from src.data.dataset_iiit5k import IIIT5KDataset

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Set up image transforms from config
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config['model']['input_height'], config['model']['input_width'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['data']['image_preprocessing']['normalize']['mean'],
                         std=config['data']['image_preprocessing']['normalize']['std'])
])

# Instantiate dataset and dataloader
dataset = IIIT5KDataset(
    img_dir=config['data']['train_data_path'],
    anno_file=config['data']['annotation_file'],
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Instantiate model
model = CRNN(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
ctc_loss = torch.nn.CTCLoss(blank=config['charset']['blank_token'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Charset encoding utility
def encode_label(label, charset):
    char_to_idx = {c: i+1 for i, c in enumerate(charset)}  # +1 if blank=0
    return [char_to_idx[c] for c in label if c in char_to_idx]

# Training loop skeleton
for epoch in range(config['training']['epochs']):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        # Prepare targets for CTC Loss
        targets = [torch.tensor(encode_label(l, config['charset']['characters'])) for l in labels]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # [batch, seq_len, num_classes]
        # CTC expects [seq_len, batch, num_classes]
        outputs = outputs.permute(1, 0, 2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)

        # Compute loss
        loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

print("Training complete!")

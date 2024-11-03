import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn
import logging
from src.model import RotateLayer
from src.dataset import RotateDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/training.log")],
)

# Dataset and DataLoader
dataset = RotateDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer
model = RotateLayer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 10
best_loss = float("inf")
best_model_path = "./weights/best_model.pt"

with Progress(
    TextColumn("[bold blue]Training..."),
    BarColumn(),
    TextColumn("[progress.description]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
) as progress:
    for epoch in range(num_epochs):
        task = progress.add_task(
            f"Epoch {epoch + 1}/{num_epochs}",
            total=len(dataloader),
        )
        running_loss = 0.0

        for i, (input_image, target_image) in enumerate(dataloader):
            input_image, target_image = input_image.to(device), target_image.to(device)

            optimizer.zero_grad()

            output = model(input_image)

            loss = criterion(output, target_image)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress.update(task, advance=1)
            progress.tasks[
                task
            ].description = f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}"

            if (i + 1) % 100 == 0:
                logging.info(
                    f"Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}",
                )

        epoch_loss = running_loss / len(dataloader)
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}",
        )

        progress.update(
            task,
            description=f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {epoch_loss:.4f}",
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.jit.script(model).save(best_model_path)
            logging.info(f"Best model saved to {best_model_path}")

print("Training completed.")

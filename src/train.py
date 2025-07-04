from src.utils import save_model
from src.evaluate import evaluate
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device='cpu',
          patience=10, checkpoint_path='checkpoint.pth'):
    best_val_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    t = tqdm(range(epochs))
    for epoch in t:
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        
        t.set_postfix_str(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, checkpoint_path.replace('.pth', '_acc.pth'))

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
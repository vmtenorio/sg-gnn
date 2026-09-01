import logging
import os
import uuid
import numpy as np
import torch

def train(model, x, edge_index, y, optimizer, criterion, mask, fw_kwargs):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index=edge_index, **fw_kwargs)
    loss = criterion(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, x, edge_index, y, criterion, mask, fw_kwargs):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index=edge_index, **fw_kwargs)
        loss = criterion(out[mask], y[mask])

        pred = out.argmax(dim=1)
        acc = (pred[mask] == y[mask]).float().mean()
    return loss, acc

def train_model(model, x, edge_index, y, train_mask, val_mask, test_mask, fw_kwargs, idx, lr, wd, epochs, patience, verb=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()

    if idx is not None:
        train_mask = train_mask[:,idx]
        val_mask = val_mask[:,idx]
        test_mask = test_mask[:,idx]

    random_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}" # unique across concurrent processes

    best_val_acc = 0 # float('inf')
    epochs_no_improve = 0
    torch.save(model.state_dict(), f'best_model-{random_id}.pth') # In case it does not learn anything

    train_losses = np.zeros(epochs)
    # train_accs = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    val_accs = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    test_accs = np.zeros(epochs)
    for e in range(epochs):
        train_losses[e] = train(model, x, edge_index, y, optimizer, criterion, train_mask, fw_kwargs)
        val_losses[e], val_accs[e] = test(model, x, edge_index, y, criterion, val_mask, fw_kwargs)
        test_losses[e], test_accs[e] = test(model, x, edge_index, y, criterion, test_mask, fw_kwargs)

        if val_accs[e] > best_val_acc:
            best_val_acc = val_accs[e]
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model-{random_id}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            if verb:
                logging.info("Early stopping!")
            break
        
        if verb and (e+1) % 1 == 0:
            logging.info(f"Epoch {e+1:03d}, Train Loss: {train_losses[e]:.4f}, Val loss: {val_losses[e]:.4f} Test Loss: {test_losses[e]:.4f}, Val Acc: {val_accs[e]:.4f}, Test Acc: {test_accs[e]:.4f}")

    # Load best model
    model.load_state_dict(torch.load(f'best_model-{random_id}.pth', weights_only=True))

    os.remove(f'best_model-{random_id}.pth')

    return model, train_losses, val_losses, test_losses, val_accs, test_accs

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
val_score = 1e-5

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# Training
def train(model, train_dataloader, val_dataloader, epochs, save_path, evaluation = False):
    val_score = 1e-5
    print("\n ******* Training ... ******* \n")
    
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("="*70)
        t0_epoch, t0_batch = time.time(), time.time()
        # 1. Reset tracking
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # 2. training mode
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # 2-1. Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # 2-2. Zero out any previously calculated gradients
            model.zero_grad()
            # 2-3. Logits
            logits = model(b_input_ids, b_attn_mask)
            # 2-4. Compute Loss
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            # 2-5. Calculate gradient, Backward pass
            loss.backward()
            # 2-6. Clip the norm of the gradients to 1.0 (prevent 'exploading gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 2-7. Update parameters & learning rate
            optimizer.step()
            scheduler.step()

            # Print per 1000 STEPS
            if (step % 1000 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)


        # 3. Validation mode
        print("="*70)
        if evaluation == True:
            val_loss, val_accuracy, val_score = evaluate(model, val_dataloader, val_score, save_path)
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("="*70)
        print("\n")
    print("Training complete !")


def evaluate(model, val_dataloader, val_score, save_path):
    print("  Validation score  ")
    # 3. Validation mode
    model.eval()
    # Tracking variables
    val_accuracy, val_loss = [], []

    for batch in val_dataloader:
        # 3-1. Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 3-2. Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # 3-3. Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # 3-4. Get the predictions(Maximum)
        preds = torch.argmax(logits, dim=1).flatten()

        # 3-5. Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # 3-6. Compute the average accuracy and loss
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    # 3-7. Save best model
    if val_accuracy > val_score:
        val_score = val_accuracy
        if not os.path.isdir("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), save_path)

    return val_loss, val_accuracy, val_score


def model_predict(model, test_dataloader):
    model.eval()
    model.to(device)
    all_logits = []

    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2] # No labels

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim = 0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim = 1)  # [num_text, num_classes]

    # Indices of the maximum probability of all classes
    preds = torch.argmax(probs, dim = 1).flatten().cpu().numpy()  # [maximum_prob_indices]

    return preds  
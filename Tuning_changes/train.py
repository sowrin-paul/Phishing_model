import torch
import torch.nn as nn
import torch.optim as optim
from sympy import sequence
from torch.onnx.ops import attention
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PhishingTrainer:
    def __init__(self, model, train_loader, valid_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        self.train_loss = []
        self.valid_loss = []
        self.valid_accuracy = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            sequences = batch['sequence'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            outputs, attention_weights = self.model(sequences, masks)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_prediction = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validation"):
                sequences = batch['sequence'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs, attention_weight = self.model(sequences, masks)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                prediction = (outputs > 0.5).float()
                all_prediction.extend(prediction.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.valid_loader)
        accuracy = accuracy_score(all_labels, all_prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_prediction, average='binary'
        )

        try:
            auc = roc_auc_score(all_labels, all_prediction)
        except ValueError:
            auc = 0.0

        return avg_loss, accuracy, precision, recall, f1, auc

    def train(self, num_epochs=20, save_path='best_model.pth', patience=3):
        best_val_loss = float('inf')
        patience_counter = 0  # Counter for early stopping

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch()

            val_loss, accuracy, precision, recall, f1, auc = self.validate()

            # lr scheduling
            self.scheduler.step(val_loss)

            # Save metrics
            self.train_loss.append(train_loss)
            self.valid_loss.append(val_loss)
            self.valid_accuracy.append(accuracy)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"AUC: {auc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'accuracy': accuracy
                }, save_path)
                print(f"New best model saved with val_loss: {val_loss:.4f}")
                patience_counter = 0  # Reset patience counter when the best model is saved
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.train_loss, label='Train Loss')
        ax1.plot(self.valid_loss, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        #
        ax2.plot(self.valid_accuracy, label='Val Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def early_stopping(self, val_loss, patience=3):
        if len(self.valid_loss) > patience and val_loss > min(self.valid_loss[-patience:]):
            print("Early stopping triggered!")
            return True
        return False

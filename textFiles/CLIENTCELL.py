# CLIENT CELL
import torch.nn as nn
import torch.optim as optim
from flwr.client import NumPyClient

def get_weights(model):
    """Extract model weights as a list of NumPy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    """Set model weights from a list of NumPy arrays"""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

class TrustMCClient(NumPyClient):
    """
    TRUST_MCNet federated learning client implementation
    
    This client handles:
    - Local model training
    - Parameter exchange with the server
    - Model evaluation and metrics reporting
    """
    def __init__(self, model, train_loader, test_loader, client_id=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Track training history
        self.train_losses = []
        self.test_accuracies = []
        
        # Track metrics for trust evaluation
        self.last_accuracy = 0.0
        self.round_number = 0
        
    def get_parameters(self, config):
        """Return current model parameters as a list of NumPy arrays"""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """Train the model on local data and return updated parameters"""
        # Update model with received parameters
        set_weights(self.model, parameters)
        
        # Track rounds
        self.round_number += 1
        
        # Training mode
        self.model.train()
        
        # Track metrics
        batch_losses = []
        
        # Train for one epoch
        for X, y in self.train_loader:
            # Zero gradients
            self.optim.zero_grad()
            
            # Forward pass
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            self.optim.step()
            
            # Store batch loss
            batch_losses.append(loss.item())
        
        # Calculate training loss
        train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        self.train_losses.append(train_loss)
        
        # Calculate additional metrics for trust evaluation
        additional_metrics = self._calculate_trust_metrics()
        
        # Add the loss to the metrics
        additional_metrics["train_loss"] = train_loss
        
        return get_weights(self.model), len(self.train_loader.dataset), additional_metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data"""
        # Update model with received parameters
        set_weights(self.model, parameters)
        
        # Evaluation mode
        self.model.eval()
        
        # Tracking metrics
        correct, total = 0, 0
        loss_total = 0.0
        predictions = []
        true_labels = []
        
        # No gradient tracking for evaluation
        with torch.no_grad():
            for X, y in self.test_loader:
                # Forward pass
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                
                # Track loss
                loss_total += loss.item()
                
                # Calculate accuracy
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                # Store predictions and labels for F1 score calculation
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(y.cpu().numpy())
        
        # Calculate metrics
        test_loss = loss_total / len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        # Store current accuracy for future trust calculations
        delta_accuracy = accuracy - self.last_accuracy
        self.last_accuracy = accuracy
        self.test_accuracies.append(accuracy)
        
        # Return evaluation results
        metrics = {
            "loss": test_loss,
            "accuracy": accuracy,
            "delta_accuracy": delta_accuracy
        }
        
        return test_loss, total, metrics
    
    def _calculate_trust_metrics(self):
        """Calculate metrics used for trust evaluation"""
        metrics = {}
        
        # 1. Cosine similarity will be calculated at server side
        
        # 2. Prediction entropy (uncertainty)
        entropy = self._calculate_prediction_entropy()
        metrics["ent"] = entropy
        
        # 3. Reputation from history
        if len(self.test_accuracies) > 1:
            # Use improvement in accuracy as reputation
            acc_deltas = [self.test_accuracies[i] - self.test_accuracies[i-1] 
                          for i in range(1, len(self.test_accuracies))]
            reputation = sum(acc_deltas) / len(acc_deltas) if acc_deltas else 0
            # Normalize to [0, 1]
            reputation = (reputation + 0.2) / 0.4  # Assuming deltas in [-0.2, 0.2]
            reputation = max(0, min(1, reputation))  # Clip to [0, 1]
        else:
            reputation = 0.5  # Initial neutral reputation
        
        metrics["rep"] = reputation
        
        return metrics
    
    def _calculate_prediction_entropy(self):
        """Calculate average prediction entropy on test data"""
        if not hasattr(self, 'test_loader') or not self.test_loader:
            return 0.5  # Default value
        
        self.model.eval()
        entropies = []
        
        with torch.no_grad():
            # Sample a subset of test data for efficiency
            for i, (X, _) in enumerate(self.test_loader):
                if i >= 3:  # Limit to first 3 batches for efficiency
                    break
                    
                # Get predicted probabilities
                logits = self.model(X)
                probs = F.softmax(logits, dim=1)
                
                # Calculate entropy: -sum(p*log(p))
                log_probs = torch.log(probs + 1e-10)  # Avoid log(0)
                batch_entropy = -torch.sum(probs * log_probs, dim=1)
                
                # Normalize by log(num_classes) to get [0,1] range
                batch_entropy = batch_entropy / np.log(logits.size(1))
                
                entropies.extend(batch_entropy.cpu().numpy())
        
        # Average entropy across all samples
        mean_entropy = np.mean(entropies) if entropies else 0.5
        
        # Transform so that lower entropy gives higher trust score
        # 1 - normalized entropy gives a [0,1] score where 1 is certain prediction
        return 1 - mean_entropy
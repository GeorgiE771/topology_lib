import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    """
    Простейший автоэнкодер для 2D->(код)->2D:
    - Encoder: 2D -> hidden -> code_dim
    - Decoder: code_dim -> hidden -> 2D
    """
    def __init__(self, input_dim=2, hidden_dim=16, code_dim=2):
        super(SimpleAutoencoder, self).__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim)
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)       # "сжимаем" вход в код
        out = self.decoder(z)     # восстанавливаем
        return out, z
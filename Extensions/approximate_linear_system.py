import torch
import torch.nn as nn
import json
import argparse
from Linear_Systems_Regression.simple_model import Model

class Approximator:
    def __init__(self, A_data_json: str, A_shape_json: str, B_data_json: str, B_shape_json: str, model_file_name: str) -> None:
        self.A_data = json.loads(A_data_json)
        self.A_shape = json.loads(A_shape_json)
        self.B_data = json.loads(B_data_json)
        self.B_shape = json.loads(B_shape_json)
        self.model_file_name = model_file_name
        self.model = None

    def load_model(self) -> nn.Module:
        model_path = f'Extensions/Linear_Systems_Regression/{self.model_file_name}'
        model = Model(self.A_shape[1])  # Assuming A_shape[1] gives the number of columns in A
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def run_inference(self) -> torch.Tensor:
        A_tensor = torch.tensor(self.A_data, dtype=torch.float).view(self.A_shape)
        B_tensor = torch.tensor(self.B_data, dtype=torch.float).view(self.B_shape)

        combined_tensor = torch.cat((A_tensor.view(-1), B_tensor.view(-1))).unsqueeze(0)
        output_tensor = self.model(combined_tensor)
        return output_tensor

    def main(self) -> None:
        self.model = self.load_model()
        output_tensor = self.run_inference()
        print(' '.join(map(str, output_tensor.squeeze().tolist())))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('A_data')
    parser.add_argument('A_shape')
    parser.add_argument('B_data')
    parser.add_argument('B_shape')
    parser.add_argument('model_file_name')

    args = parser.parse_args()
    approximator = Approximator(args.A_data, args.A_shape, args.B_data, args.B_shape, args.model_file_name)
    approximator.main()

import os
import csv
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from vgg_net import VGGNet
from vgg_dataset import CustomDataset


classes = [
    "PASS",
    "FIBER",
    "REJECT",
    "SCRATCH",
    "STAIN",
]

channels = ["ch0", "ch1", "ch2", "ch3", "ch4"]

transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40),
        transforms.ToTensor(),
    ]
)

train_dataset = CustomDataset(
    root_dir="C:\\Users\\IntekPlus\\Desktop\\dataset\\BU1_Classification(20220922)_include_test",
    classes=classes,
    channels=channels,
    transform=transform,
    mode="Train",
)
valid_dataset = CustomDataset(
    root_dir="C:\\Users\\IntekPlus\\Desktop\\dataset\\BU1_Classification(20220922)_include_test",
    classes=classes,
    channels=channels,
    transform=transform,
    mode="Valid",
)
test_dataset = CustomDataset(
    root_dir="C:\\Users\\IntekPlus\\Desktop\\dataset\\BU1_Classification(20220922)_include_test",
    classes=classes,
    channels=channels,
    transform=transform,
    mode="Test",
)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    model = VGGNet(num_classes=5).to(device)

    # best_model_path = "best_model.pth"
    # try:
    #     model.load_state_dict(torch.load(best_model_path))

    # except FileNotFoundError:
    #     print(f"저장된 모델이 없습니다. 처음부터 학습을 시작합니다.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    epochs = 30

    best_accuracy = 0.0
    best_model_path = "best_model_224.pth"

    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct_train = 0
    #     total_train = 0

    #     for i, (inputs, labels) in enumerate(train_loader):
    #         optimizer.zero_grad()

    #         inputs, labels = inputs.to(device), labels.to(device)

    #         labels = torch.argmax(labels, dim=1)

    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         _, predicted = torch.max(outputs, 1)
    #         total_train += labels.size(0)
    #         correct_train += (predicted == labels).sum().item()

    #     avg_train_loss = running_loss / len(train_loader)
    #     train_accuracy = 100 * correct_train / total_train
    #     print(
    #         f"Epoch [{epoch + 1}/{epochs}], "
    #         f"Train Loss: {avg_train_loss:.3f}, "
    #         f"Train Accuracy: {train_accuracy:.4f}%"
    #     )

    #     model.eval()
    #     valid_loss = 0.0
    #     correct_valid = 0
    #     total_valid = 0
    #     with torch.no_grad():
    #         for inputs, labels in valid_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)

    #             labels = torch.argmax(labels, dim=1)

    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             valid_loss += loss.item()

    #             _, predicted = torch.max(outputs, 1)
    #             total_valid += labels.size(0)
    #             correct_valid += (predicted == labels).sum().item()

    #     avg_valid_loss = valid_loss / len(valid_loader)
    #     valid_accuracy = 100 * correct_valid / total_valid
    #     print(
    #         f"Validation Loss: {avg_valid_loss:.3f}, "
    #         f"Validation Accuracy: {valid_accuracy:.4f}%"
    #     )

    #     if valid_accuracy > best_accuracy:
    #         best_accuracy = valid_accuracy
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f"새로운 최고 정확도: {best_accuracy:.4f}%, 모델 저장 완료.")

    # print("Finished Training")

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    csv_data = [["Image_Name", "True_Label", "Predicted_Label"]]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = torch.argmax(labels, dim=1)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            batch_size = inputs.size(0)
            for i in range(batch_size):
                image_name = test_dataset.data["image"][
                    total_test - batch_size + i
                ][0]
                true_label = labels[i].item()
                predicted_label = predicted[i].item()

                csv_data.append(
                    [os.path.basename(image_name), true_label, predicted_label]
                )

    csv_file_path = "test_results.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(
        f"Test Loss: {avg_test_loss:.3f}, "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )
    print(f"Test results saved to {csv_file_path}")

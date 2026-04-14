import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('/home/ssssss/1yolo/Ablation_Results/Exp1_BASELINE/weights/last.pt')
    model.train(
        resume=True,
        )

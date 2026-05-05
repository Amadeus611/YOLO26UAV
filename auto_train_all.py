import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
import torch  # type: ignore
from ultralytics import YOLO


# =========================================================
# YOLO26s + UAVDT 消融实验
# =========================================================

# --- 所有实验共享参数 ---
COMMON = dict(
    data="UAVDT.yaml",
    project="/home/ssssss/1yolo/Ablation_Results",
    weight="yolo26s.pt",
    imgsz=640,
    batch=32,
    epochs=150,
    device=0,
    workers=8,
    val=True,
    plots=True,
    save=True,
    amp=True,
    cache=False,
    # 优化器
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,
    warmup_epochs=10,
    # 损失
    box=7.5,
    cls=0.8,
    dfl=1.5,
    cls_pw=0.5,
    # 训练策略
    patience=70,
    # 数据增强 (航拍适配)
    mosaic=1.0,
    close_mosaic=20,
    mixup=0.0,
    copy_paste=0.0,
    degrees=25.0,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    erasing=0.1,
)


def main():
    experiments = [
        # =====================================================
        # 快速验证：基准 vs 结构改进 vs 全部改进
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
            "name": "Exp01_Baseline",
        },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab02_tae.yaml",
        #     "name": "Exp02_TAE",
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab03_csff.yaml",
        #     "name": "Exp03_CSFF",
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab04_uavdetect.yaml",
        #     "name": "Exp04_UAVDetect",
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
        #     "name": "Exp05_DRFL",
        #     "use_uav_loss": True,
        #     "uav_noise_beta": 0.05,
        #     "uav_loss_weight": 0.05,
        #     "tail_class_boost": 1.15,
        # },
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab07_csff_uavdetect.yaml",
            "name": "Exp06_CSFF_UAVDetect",
        },
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab08_tae_csff_uavdetect.yaml",
            "name": "Exp07_Full",
            "use_uav_loss": True,
            "uav_noise_beta": 0.05,
            "uav_loss_weight": 0.05,
            "tail_class_boost": 1.15,
        },

        # =====================================================
        # CSFF 子模块消融（对照 Exp03）
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab05_csff_no_spatial.yaml",
        #     "name": "Exp08_CSFF_NoSpatial",
        # },

        # =====================================================
        # UAVDetect 子模块消融（对照 Exp04）
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_uavdetect_no_refine.yaml",
        #     "name": "Exp09_UAVDetect_NoRefine",
        # },

        # =====================================================
        # DRFL 子模块消融（对照 Exp05）
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
        #     "name": "Exp10_DRFL_FocalOnly",
        #     "use_uav_loss": True,
        #     "uav_noise_beta": 0.0,
        #     "uav_loss_weight": 0.05,
        #     "tail_class_boost": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
        #     "name": "Exp11_DRFL_FocalNoise",
        #     "use_uav_loss": True,
        #     "uav_noise_beta": 0.05,
        #     "uav_loss_weight": 0.05,
        #     "tail_class_boost": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
        #     "name": "Exp12_DRFL_FocalTail",
        #     "use_uav_loss": True,
        #     "uav_noise_beta": 0.0,
        #     "uav_loss_weight": 0.05,
        #     "tail_class_boost": 1.15,
        # },
    ]

    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"  实验 {i + 1}/{len(experiments)}: {exp['name']}")
        print(f"  配置文件: {exp['yaml']}")
        print(f"{'=' * 60}\n")

        cfg = dict(COMMON, name=exp["name"])

        # DRFL 相关参数
        cfg["use_uav_loss"] = exp.get("use_uav_loss", False)
        cfg["uav_loss_gamma"] = 1.0
        cfg["uav_loss_alpha"] = 0.5
        cfg["uav_noise_beta"] = exp.get("uav_noise_beta", 0.0)
        cfg["uav_loss_weight"] = exp.get("uav_loss_weight", 0.0)
        cfg["uav_tail_temperature"] = 1.0
        cfg["uav_loss_trunc"] = 3.0
        cfg["tail_class_boost"] = exp.get("tail_class_boost", 1.0)

        model = YOLO(exp["yaml"]).load(COMMON["weight"])
        model.train(**cfg)
        torch.cuda.empty_cache()

    print("\n  所有消融实验已全部执行完毕！")


if __name__ == "__main__":
    main()

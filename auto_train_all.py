import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")

from ultralytics import YOLO
import torch  # type: ignore


def main():
    # =========================================================
    # 消融实验任务列表
    # ---------------------------------------------------------
    # 当前脚本用于第二轮快速定位：
    # 1. 停用 P2/full 堆叠链路
    # 2. 回到强 baseline，只挂一个最小改动
    # 3. DRFL 只作为 BCE 的辅助项，不再替代 BCE
    #
    # 已有可对比基线：
    #   Exp01_A0_YOLO26s_Baseline
    #
    # 本轮激活实验：
    #   Exp15: baseline + TAE only
    #   Exp16: baseline + CSFF residual only
    #   Exp17: baseline + UAVDetect only
    #   Exp18: baseline + auxiliary DRFL loss only
    # =========================================================

    experiments = [
        # =====================================================
        # Baseline reference. 已经跑过，默认不重复跑。
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
        #     "name": "Exp15_A0_Baseline_Rerun_MinQ50",
        #     "batch": 8,
        #     "cls_gain": 0.8,
        #     "use_uav_loss": False,
        #     "uav_loss_gamma": 1.0,
        #     "uav_loss_alpha": 0.5,
        #     "uav_noise_beta": 0.0,
        #     "uav_loss_weight": 0.0,
        #     "uav_tail_temperature": 1.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },

        # =====================================================
        # Minimal structure ablations on top of baseline.
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab11_baseline_tae.yaml",
            "name": "Exp15_Baseline_TAE_MinQ50",
            "batch": 8,
            "cls_gain": 0.8,
            "use_uav_loss": False,
            "uav_loss_gamma": 1.0,
            "uav_loss_alpha": 0.5,
            "uav_noise_beta": 0.0,
            "uav_loss_weight": 0.0,
            "uav_tail_temperature": 1.0,
            "tail_class_boost": 1.0,
            "use_weighted_sampler": False,
            "sample_tail_gain": 1.0,
        },
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab12_baseline_csff.yaml",
            "name": "Exp16_Baseline_CSFF_MinQ50",
            "batch": 8,
            "cls_gain": 0.8,
            "use_uav_loss": False,
            "uav_loss_gamma": 1.0,
            "uav_loss_alpha": 0.5,
            "uav_noise_beta": 0.0,
            "uav_loss_weight": 0.0,
            "uav_tail_temperature": 1.0,
            "tail_class_boost": 1.0,
            "use_weighted_sampler": False,
            "sample_tail_gain": 1.0,
        },
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab13_baseline_uavdetect.yaml",
            "name": "Exp17_Baseline_UAVDetect_MinQ50",
            "batch": 8,
            "cls_gain": 0.8,
            "use_uav_loss": False,
            "uav_loss_gamma": 1.0,
            "uav_loss_alpha": 0.5,
            "uav_noise_beta": 0.0,
            "uav_loss_weight": 0.0,
            "uav_tail_temperature": 1.0,
            "tail_class_boost": 1.0,
            "use_weighted_sampler": False,
            "sample_tail_gain": 1.0,
        },

        # =====================================================
        # Loss-only ablation on baseline.
        # DRFL is an auxiliary regularizer: BCE + 0.05 * DRFL.
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
            "name": "Exp18_Baseline_AuxDRFL_MinQ50",
            "batch": 8,
            "cls_gain": 0.8,
            "use_uav_loss": True,
            "uav_loss_gamma": 1.0,
            "uav_loss_alpha": 0.5,
            "uav_noise_beta": 0.05,
            "uav_loss_weight": 0.05,
            "uav_tail_temperature": 1.0,
            "tail_class_boost": 1.15,
            "use_weighted_sampler": False,
            "sample_tail_gain": 1.0,
        },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"开始执行第 {i + 1}/{len(experiments)} 组实验: {exp['name']}")
        print(f"模型结构 YAML: {exp['yaml']}")
        print(f"UAV 辅助损失开关: {exp['use_uav_loss']}")
        print(f"UAV 辅助损失权重: {exp['uav_loss_weight']}")
        print(f"尾类 loss boost 倍率: {exp['tail_class_boost']}")
        print(f"尾类加权采样开关: {exp['use_weighted_sampler']}")
        print(f"尾类采样增益倍率: {exp['sample_tail_gain']}")
        print(f"Batch Size: {exp['batch']}")
        print(f"{'=' * 60}\n")

        # 实例化模型：加载 YAML 网络结构，并加载 YOLO26s 预训练权重。
        model = YOLO(exp["yaml"]).load("yolo26s.pt")

        # 启动训练。
        model.train(
            # -------------------------------------------------
            # 实验核心变量
            # -------------------------------------------------
            data="ultralytics/cfg/datasets/EVD4UAV.yaml",
            name=exp["name"],
            batch=exp["batch"],
            project="Ablation_Result",

            # -------------------------------------------------
            # 基础训练设置
            # -------------------------------------------------
            epochs=50,
            imgsz=960,
            device=0,
            workers=8,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=0,
            amp=True,
            cache=False,

            # -------------------------------------------------
            # 无人机航拍增强设置
            # -------------------------------------------------
            multi_scale=0.0,
            mosaic=1.0,
            close_mosaic=15,
            mixup=0.05,
            copy_paste=0.0,
            degrees=12.0,
            translate=0.12,
            scale=0.75,
            flipud=0.3,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,

            # -------------------------------------------------
            # 损失函数消融开关
            # -------------------------------------------------
            cls=exp["cls_gain"],
            cls_pw=0.5,
            use_uav_loss=exp["use_uav_loss"],
            uav_loss_gamma=exp["uav_loss_gamma"],
            uav_loss_alpha=exp["uav_loss_alpha"],
            uav_noise_beta=exp["uav_noise_beta"],
            uav_loss_weight=exp["uav_loss_weight"],
            uav_tail_temperature=exp["uav_tail_temperature"],
            uav_loss_trunc=3.0,
            tail_class_boost=exp["tail_class_boost"],
            manual_class_weights=[],

            # -------------------------------------------------
            # bus/truck 参数策略开关，不作为创新点
            # -------------------------------------------------
            use_weighted_sampler=exp["use_weighted_sampler"],
            sample_weight_power=0.6,
            sample_tail_gain=exp["sample_tail_gain"],

            # -------------------------------------------------
            # 验证与可视化
            # -------------------------------------------------
            val=True,
            plots=True,
        )

        # 每组实验结束后清空显存缓存，避免下一组实验显存残留。
        torch.cuda.empty_cache()

    print("\n所有消融实验已全部顺序执行完毕。")


if __name__ == "__main__":
    main()

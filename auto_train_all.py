import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")

from ultralytics import YOLO
import torch  # type: ignore


def main():
    # =========================================================
    # 消融实验任务列表
    # ---------------------------------------------------------
    # Table 1: 主创新点逐步叠加消融
    #   Exp01: 原始 YOLO26s baseline
    #   Exp02: + P2 小目标检测分支
    #   Exp03: + P2 + TAE 纹理感知增强
    #   Exp04: + P2 + CSFF 跨尺度选择性融合
    #   Exp05: + P2 + TAE + CSFF，仍使用原始 Detect 头
    #
    # Table 2: 副创新点检测头消融
    #   Exp06: + UAVDetect 检测头
    #
    # Table 3: 损失函数消融，结构固定为完整 UAVDetect
    #   Exp07: BCE
    #   Exp08: Focal
    #   Exp09: Robust Focal
    #   Exp10: Dynamic Robust Long-tail Loss（论文最终方案）
    #
    # Table 4: bus/truck 参数策略消融，不作为论文创新点
    #   Exp11: 不使用尾类策略
    #   Exp12: 只使用尾类加权采样
    #   Exp13: 只使用尾类 loss boost
    #   Exp14: 加权采样 + 尾类 loss boost
    #
    # 注意：
    #   bus/truck 的类别索引不写在本训练脚本里。
    #   类别映射统一写在 ultralytics/cfg/default.yaml 和 ultralytics/cfg/uav_vehicle.yaml。
    # =========================================================

    experiments = [
        # =====================================================
        # Table 1: 主创新点逐步叠加消融
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab01_baseline.yaml",
            "name": "Exp01_A0_YOLO26s_Baseline",
            "batch": 8,
            "use_uav_loss": False,          # 是否启用本文设计的 UAV 鲁棒长尾分类损失
            "uav_noise_beta": 0.0,          # 噪声标签抑制强度，0 表示不抑制
            "tail_class_boost": 1.0,        # 尾类 loss 放大系数，1.0 表示不放大
            "use_weighted_sampler": False,  # 是否启用尾类图片加权采样
            "sample_tail_gain": 1.0,        # 尾类图片采样权重放大倍数
        },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab02_p2.yaml",
        #     "name": "Exp02_A1_P2",
        #     "batch": 8,
        #     "use_uav_loss": False,
        #     "uav_noise_beta": 0.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab03_p2_tae.yaml",
        #     "name": "Exp03_A2_P2_TAE",
        #     "batch": 8,
        #     "use_uav_loss": False,
        #     "uav_noise_beta": 0.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab04_p2_csff.yaml",
        #     "name": "Exp04_A3_P2_CSFF",
        #     "batch": 8,
        #     "use_uav_loss": False,
        #     "uav_noise_beta": 0.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab05_p2_tae_csff_detect.yaml",
        #     "name": "Exp05_A4_P2_TAE_CSFF_Detect",
        #     "batch": 8,
        #     "use_uav_loss": False,
        #     "uav_noise_beta": 0.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },

        # # =====================================================
        # # Table 2: 副创新点 UAVDetect 检测头消融
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp06_H1_UAVDetect_BCE",
        #     "batch": 8,
        #     "use_uav_loss": False,
        #     "uav_noise_beta": 0.0,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },

        # # =====================================================
        # # Table 3: 损失函数消融，模型结构固定为完整 UAVDetect
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp07_L0_BCE",
        #     "batch": 8,
        #     "use_uav_loss": False,          # 原始 BCE 分类损失
        #     "uav_loss_gamma": 2.0,          # Focal 指数，此组不用 UAV loss，仅占位
        #     "uav_loss_alpha": 0.25,         # 正负样本平衡系数，此组不用 UAV loss，仅占位
        #     "uav_noise_beta": 0.0,          # 噪声抑制关闭
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp08_L1_Focal",
        #     "batch": 8,
        #     "use_uav_loss": True,           # 启用 UAV loss 框架，用 gamma/alpha 表现 Focal loss
        #     "uav_loss_gamma": 2.0,          # 越大越关注难分类样本
        #     "uav_loss_alpha": 0.25,         # 正样本权重，缓解前景/背景不平衡
        #     "uav_noise_beta": 0.0,          # 设为 0，仅验证 Focal，不做噪声标签抑制
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp09_L2_RobustFocal",
        #     "batch": 8,
        #     "use_uav_loss": True,
        #     "uav_loss_gamma": 2.0,
        #     "uav_loss_alpha": 0.25,
        #     "uav_noise_beta": 0.2,          # 噪声标签抑制强度，降低疑似错标样本的梯度影响
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        {
            "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
            "name": "Exp10_L3_DynamicRobustLongTail",
            "batch": 8,
            "use_uav_loss": True,
            "uav_loss_gamma": 2.0,
            "uav_loss_alpha": 0.30,
            "uav_noise_beta": 0.2,
            "tail_class_boost": 1.35,       # 放大配置中尾类 bus/truck 的分类 loss
            "use_weighted_sampler": False,
            "sample_tail_gain": 1.0,
        },

        # =====================================================
        # Table 4: bus/truck 参数策略消融，不作为论文创新点
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp11_T0_NoTailStrategy",
        #     "batch": 8,
        #     "use_uav_loss": True,
        #     "uav_loss_gamma": 2.0,
        #     "uav_loss_alpha": 0.25,
        #     "uav_noise_beta": 0.2,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp12_T1_WeightedSampler",
        #     "batch": 8,
        #     "use_uav_loss": True,
        #     "uav_loss_gamma": 2.0,
        #     "uav_loss_alpha": 0.25,
        #     "uav_noise_beta": 0.2,
        #     "tail_class_boost": 1.0,
        #     "use_weighted_sampler": True,   # 启用配置中尾类 bus/truck 的图片加权采样
        #     "sample_tail_gain": 1.4,        # 含 bus/truck 图片被抽到的概率放大 1.4 倍
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp13_T2_TailLossBoost",
        #     "batch": 8,
        #     "use_uav_loss": True,
        #     "uav_loss_gamma": 2.0,
        #     "uav_loss_alpha": 0.30,
        #     "uav_noise_beta": 0.2,
        #     "tail_class_boost": 1.35,       # 只放大配置中尾类 bus/truck 对分类 loss 的贡献
        #     "use_weighted_sampler": False,
        #     "sample_tail_gain": 1.0,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s_ab06_full_uavdetect.yaml",
        #     "name": "Exp14_T3_FullTailStrategy",
        #     "batch": 8,
        #     "use_uav_loss": True,
        #     "uav_loss_gamma": 2.0,
        #     "uav_loss_alpha": 0.30,
        #     "uav_noise_beta": 0.2,
        #     "tail_class_boost": 1.35,
        #     "use_weighted_sampler": True,
        #     "sample_tail_gain": 1.4,
        # },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"开始执行第 {i + 1}/{len(experiments)} 组实验: {exp['name']}")
        print(f"模型结构 YAML: {exp['yaml']}")
        print(f"UAV 鲁棒长尾损失开关: {exp['use_uav_loss']}")
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
            data="ultralytics/cfg/datasets/EVD4UAV.yaml",  # EVD4UAV 数据集配置文件
            name=exp["name"],                     # 当前实验名称，结果会保存到 project/name
            batch=exp["batch"],                   # 每张 GPU 的 batch size，显存不够就调小
            project="/home/ssssss/1yolo/Ablation_Results",  # 实验结果保存目录

            # -------------------------------------------------
            # 基础训练设置
            # -------------------------------------------------
            epochs=100,                           # 训练轮数，论文正式实验建议所有组保持一致
            imgsz=1024,                            # 输入图像尺寸，航拍小目标建议用较高分辨率
            device=0,                             # 使用第 0 张 GPU
            workers=8,                            # dataloader 线程数
            optimizer="AdamW",                    # 优化器，AdamW 对改进结构通常更稳
            lr0=0.001,                           # 初始学习率
            lrf=0.1,                              # 最终学习率比例，最终 lr = lr0 * lrf
            momentum=0.937,                       # 优化器动量参数
            weight_decay=0.0005,                  # 权重衰减，抑制过拟合
            warmup_epochs=3.0,                    # warmup 轮数，训练初期稳定梯度
            patience=0,                           # 关闭早停，保证各消融组训练轮数公平
            amp=True,                            # 是否启用混合精度，False 更稳定但速度稍慢
            cache=False,                          # 是否缓存数据，机械硬盘空间/内存不够时建议 False

            # -------------------------------------------------
            # 无人机航拍增强设置
            # -------------------------------------------------
            multi_scale=0.0,                      # 多尺度训练幅度，增强尺度变化鲁棒性
            mosaic=1.0,                           # Mosaic 增强概率，提升密集小目标学习
            close_mosaic=15,                      # 最后 15 个 epoch 关闭 Mosaic，利于收敛
            mixup=0.05,                           # MixUp 概率，轻微提升泛化，过大可能伤害小目标
            copy_paste=0.0,                       # Copy-Paste 概率，检测任务此处默认关闭
            degrees=12.0,                         # 随机旋转角度，模拟无人机视角变化
            translate=0.12,                       # 随机平移幅度，模拟画面边缘截断
            scale=0.75,                           # 随机缩放幅度，模拟不同飞行高度
            flipud=0.3,                           # 上下翻转概率，适配俯视航拍方向不固定
            fliplr=0.5,                           # 左右翻转概率
            hsv_h=0.015,                          # 色调增强幅度
            hsv_s=0.5,                            # 饱和度增强幅度
            hsv_v=0.3,                            # 亮度增强幅度

            # -------------------------------------------------
            # 损失函数消融开关
            # -------------------------------------------------
            cls=0.8,                              # 分类 loss 权重
            cls_pw=0.5,                           # 原始分类正样本权重
            use_uav_loss=exp["use_uav_loss"],     # 是否启用 Dynamic Robust Long-tail Loss
            uav_loss_gamma=exp.get("uav_loss_gamma", 2.0),        # Focal 难样本聚焦指数
            uav_loss_alpha=exp.get("uav_loss_alpha", 0.25),       # 正负样本平衡系数
            uav_noise_beta=exp["uav_noise_beta"],                 # 噪声标签抑制强度
            uav_tail_temperature=1.1,             # 尾类权重温度，控制 tail boost 的平滑程度
            uav_loss_trunc=3.0,                   # 噪声抑制下限截断，避免权重过小
            tail_class_boost=exp["tail_class_boost"],             # 配置中尾类的分类 loss 放大系数
            manual_class_weights=[],              # 手动类别权重，空列表表示不用

            # -------------------------------------------------
            # bus/truck 参数策略开关，不作为创新点
            # -------------------------------------------------
            use_weighted_sampler=exp["use_weighted_sampler"],     # 是否启用样本加权采样
            sample_weight_power=0.6,              # 类别频次反比权重指数，越大越偏向少样本类
            sample_tail_gain=exp["sample_tail_gain"],             # 含尾类图片的额外采样增益

            # -------------------------------------------------
            # 验证与可视化
            # -------------------------------------------------
            val=True,                             # 每轮训练后执行验证
            plots=True,                           # 保存 PR 曲线、混淆矩阵等图
        )

        # 每组实验结束后清空显存缓存，避免下一组实验显存残留。
        torch.cuda.empty_cache()

    print("\n所有消融实验已全部顺序执行完毕。")


if __name__ == "__main__":
    main()

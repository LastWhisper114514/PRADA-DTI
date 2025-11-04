config = {
    "DRUG": {
        "NODE_IN_FEATS": 75,  # 药物输入特征维度
        "NODE_IN_EMBEDDING": 128,  # 药物嵌入维度
        "HIDDEN_LAYERS": [128, 128, 128],  # 药物模型的隐藏层维度
        "PADDING": True,  # 药物的填充方式（如果有需要的话）
    },
    "PROTEIN": {
        "EMBEDDING_DIM": 128,  # 蛋白质嵌入维度
        "NUM_FILTERS": [128, 128, 128],  # 蛋白质CNN的卷积核数量
        "KERNEL_SIZE": [3, 6, 9],  # 卷积核大小
        "PADDING": 1,  # 蛋白质的填充方式
    },
    "DECODER": {
        "IN_DIM": 256,  # 解码器输入维度
        "HIDDEN_DIM": 512,  # 解码器隐藏层维度
        "OUT_DIM": 128,  # 输出维度（根据任务而定，比如分类任务）
        "BINARY": 1,  # 是否为二分类任务（如果是多分类任务可以设置为 False）
    },
    "BCN": {
        "HEADS": 4,  # BCN（Bi-level Attention Network）头的数量
    }
}

import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 选择设备 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ========== 2. 加载模型 ==========
# clip.load() 返回两个东西:
#   model: CLIP 模型实例
#   preprocess: 图像的预处理管道（缩放、裁剪、归一化到 224x224）
model, preprocess = clip.load("RN50", device=device)
# ========== 3. 定义图片路径和文本 ==========
image_paths = [
    "images/img1.jpg",
    "images/img2.jpg",
    "images/img3.jpg",
    "images/img4.jpg",
]
texts = [
    "A photo of a blooming pink cherry blossom tree arching over a residential sidewalk under a clear blue sky, with petals falling in the sunlight",      # 与 img1 对应
    "A photo of two small bento cakes on a white marble table — a green matcha cake with Oreo crumbs and a blue-and-white swirled cake topped with blueberries and a fig slice",           # 与 img2 对应
    "A photo of a bouquet of purple roses wrapped in white paper, with a small white cream cake topped with a lit candle on a wooden table",     # 与 img3 对应
    "A photo of a fluffy silver kitten wearing a pearl necklace with a red pendant, sitting on a teal velvet stool",    # 与 img4 对应
]

# ========== 4. 提取图像特征 ==========
# 对每张图片做 preprocess, 然后 stack 成一个 batch
# preprocess 的步骤: Resize → CenterCrop(224) → ToTensor → Normalize
image_tensors = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
images = torch.stack(image_tensors).to(device)  # shape: [4, 3, 224, 224]

with torch.no_grad():  # 推理不需梯度，节省内存
    image_features = model.encode_image(images)

print(f"图像特征形状: {image_features.shape}")

# ========== 5. 提取文本特征 ==========
# clip.tokenize 将文本转为 token id, 最大长度 77
# 返回的 text_tokens 形状: [4, 77]
text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)

print(f"文本特征形状: {text_features.shape}")
# 预期输出: torch.Size([4, 1024])

# ========== 6. L2 归一化 ==========
# 除以每个向量的 L2 范数
# norm(dim=-1, keepdim=True) 计算最后一维的范数, 保持维度以便广播
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 验证归一化: 每个向量的范数应该接近 1.0
print(f"图像特征归一化后范数: {image_features.norm(dim=-1)}")
print(f"文本特征归一化后范数: {text_features.norm(dim=-1)}")

# ========== 7. 计算余弦相似度矩阵 ==========
# 归一化后的点积 = 余弦相似度
# image_features @ text_features.T 的 shape: [4, 4]
similarity = (image_features @ text_features.T).cpu().numpy()
print("\n相似度矩阵:")
print(similarity)

# ========== 8. 绘制热力图 ==========
plt.figure(figsize=(8, 6))
sns.heatmap(
    similarity,
    xticklabels=[f"Text {i+1}" for i in range(4)],
    yticklabels=[f"Image {i+1}" for i in range(4)],
    annot=True,       # 在格子里显示数值
    fmt=".3f",        # 保留3位小数
    cmap="YlOrRd",    # 黄-橙-红 配色
    vmin=0, vmax=1,   # 固定颜色范围
)
plt.title("CLIP Image-Text Similarity Matrix (RN50)")
plt.xlabel("Text")
plt.ylabel("Image")
plt.tight_layout()
plt.savefig("similarity_heatmap.png", dpi=150)
plt.show()

# ========== 9. 检查配对结果 ==========
predicted = similarity.argmax(axis=1)  # 每行最大值所在的列索引
print("\n配对结果:")
for i in range(4):
    is_correct = predicted[i] == i
    status = "✓ 正确" if is_correct else f"✗ 错误 (匹配到 Text {predicted[i]+1})"
    print(f"  Image {i+1} → Text {predicted[i]+1}  {status}")


texts_with_decoy = [
    "A photo of a blooming pink cherry blossom tree arching over a residential sidewalk under a clear blue sky, with petals falling in the sunlight",      # 与 img1 对应
    "A photo of two small bento cakes on a white marble table — a green matcha cake with Oreo crumbs and a blue-and-white swirled cake topped with blueberries and a fig slice",           # 与 img2 对应
    "A photo of a bouquet of purple roses wrapped in white paper, with a small white cream cake topped with a lit candle on a wooden table",     # 与 img3 对应
    "A photo of a fluffy silver kitten wearing a pearl necklace with a red pendant, sitting on a teal velvet stool",    # 与 img4 对应

     "A photo of two small bento cakes on a white marble table — a green matcha cake with Oreo crumbs and a green-and-black swirled cake topped with blueberries and a fig slice",  # 诱饵: 颜色错误
    ]

text_tokens_decoy = clip.tokenize(texts_with_decoy).to(device)
with torch.no_grad():
    text_features_decoy = model.encode_text(text_tokens_decoy)
text_features_decoy = text_features_decoy / text_features_decoy.norm(dim=-1, keepdim=True)

# 重新计算相似度: [4, 5]
similarity_decoy = (image_features @ text_features_decoy.T).cpu().numpy()

# 打印img2对所有文本的相似度
print("\n(Image 2) 对所有文本的相似度:")
for j, t in enumerate(texts_with_decoy):
    marker = " ← 正确" if j == 1 else (" ← 诱饵" if j == 4 else "")
    print(f"  {similarity_decoy[1, j]:.4f}  {t}{marker}")
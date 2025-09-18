# 🐄 Image-Based Cow Detection & Measurement (Prototype)

## 📌 Overview
This module is part of the **SECURA Prototype (SIH-25005)**.  
It handles **image-based animal type classification** with a focus on cows.  

- If a cow is detected → the system calculates **height, length, angle, etc.**  
- If no cow is detected → displays **“No cow detected”**.  
- All results are stored in **JSON format** (no database used).  

⚠️ Notes:  
- Prototype only → not optimized for production.  
- Runs locally.  
- Results are lightweight JSON files stored in repo.  

---

## 🚀 Features
- 📷 Upload an image of an animal.  
- 🧠 Detects if the image contains a **cow**.  
- 📏 Provides cow measurements:  
  - Height  
  - Length  
  - Body angle (relative to camera/ground)  
- ❌ If no cow → returns `"No cow detected"`.  
- 📄 Stores output in `outputs/` as JSON.  

---

![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spectra

## 🧩 Novelty
- **Medical-domain visual prompt injection**: This implementation evaluates prompt injection on **brain MRI scans**, a high-stakes medical setting where robustness is critical and misclassification has real-world implications.  
- **Targeted diagnostic manipulation**: Instead of measuring random prediction drift, this work tests whether injected visual text can push the model toward a **specific diagnosis (e.g., "no_tumor")**, enabling measurement of controlled adversarial influence.

## 🧠 Methodology

1. **Dataset**: Uses the [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) consisting of 3,000+ MRI images across four classes (*glioma, meningioma, pituitary, no_tumor*) as the base for injection.

2. **Injection Technique**:  
   - Overlays adversarial diagnostic text directly onto MRI images using configurable **position, font size, and opacity**.  
   - Simulates both visible and stealth prompt injections.  
   - The injected text attempts to override model reasoning by embedding a false diagnostic instruction (e.g., *"FINAL DIAGNOSIS: NO TUMOR"*).  

3. **Evaluation**:  
   - Each image is first evaluated to obtain a **baseline prediction**.  
   - The same image is then re-evaluated after injection.  
   - The pipeline runs automatically across the dataset to observe large-scale behavioral changes under attack.  

4. **Metrics**: Calculates the success rate of injections across the dataset to quantify vulnerability.  
   - **Prediction Flip Rate**: Percentage of images where prediction changes after injection.  
   - **Targeted Attack Success**: Percentage of cases where the model follows the injected diagnostic label.  
   - **Label Flip Distribution**: Tracks how predictions shift between diagnostic classes under injection.  
